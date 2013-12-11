#!/usr/bin/python
import numpy as np
import cyni
from threading import Thread
import time

import roslib, rospy
roslib.load_manifest('tf')
import tf
from sensor_msgs.msg import PointCloud2

from hd_utils.colorize import *
from hd_utils.yes_or_no import yes_or_no
from hd_utils import conversions, clouds, ros_utils as ru

from cameras import cyni_cameras
from camera_calibration import camera_calibrator
from hydra_calibration import hydra_calibrator
from gripper_calibration import gripper_calibrator
import get_marker_transforms as gmt
"""
Steps to be taken:
    1.  Calibrate cameras w.r.t. each other.
        ******** DONE
        Publish transforms.
        ******** DONE
    2.  Calibrate hydras w.r.t. cameras.
        ******** DONE
        Publish transforms.
        ******** DONE
    3.  Calibrate potentiometer.
        ******** DONE
        Calibrate gripper + angle.
        ******** DONE
        Publish transform or make graph available to get transforms.
        ******** DONE
    
"""
asus_xtion_pro_f = 544.260779961

class transform_publisher(Thread):

    def __init__(self, cameras=None):
        Thread.__init__(self)
        
        if rospy.get_name() == '/unnamed':
            rospy.init_node('calibration_node')

        self.transforms = {}
        self.ready = False
        self.rate = 30.0
        self.tf_broadcaster = tf.TransformBroadcaster()
        
        self.cameras = cameras
        self.publish_pc = False
        
        if self.cameras is not None:
            self.pc_pubs = {i:rospy.Publisher('camera%d_points'%(i+1), PointCloud2)\
                             for i in xrange(self.cameras.num_cameras)}

        

    def run (self):
        """
        Publishes the transforms stored.
        """
        while True:
            if self.publish_pc and self.cameras is not None:
                data=self.cameras.get_RGBD()
                for i,rgbd in data.items():
                    if rgbd['depth'] is None or rgbd['rgb'] is None:
                        print 'wut'
                        continue
                    camera_frame = 'camera%d_depth_optical_frame'%(i+1)
                    xyz = clouds.depth_to_xyz(rgbd['depth'], asus_xtion_pro_f)
                    pc = ru.xyzrgb2pc(xyz, rgbd['rgb'], camera_frame)
                    ar_cam = gmt.get_ar_marker_poses(None, None, pc=pc)
                    
                    self.pc_pubs[i].publish(pc)
                    marker_frame = 'ar_marker%d_camera%d'
                    for marker in ar_cam:
                        trans, rot = conversions.hmat_to_trans_rot(ar_cam[marker])
                        self.tf_broadcaster.sendTransform(trans, rot,
                                                          rospy.Time.now(),
                                                          marker_frame%(marker,i+1), 
                                                          camera_frame)
                    

            if self.ready:
                for parent, child in self.transforms:
                    trans, rot = self.transforms[parent, child]
                    self.tf_broadcaster.sendTransform(trans, rot,
                                                      rospy.Time.now(),
                                                      child, parent)
            time.sleep(1/self.rate)
            
    def set_publish_pc(self, publish_pc):
        if publish_pc and self.cameras is not None:
            self.cameras.start_streaming()
            self.publish_pc = True
        else:
            self.publish_pc = False
            

    def add_transforms(self, transforms):
        """
        Takes a list of dicts with relevant transform information.
        """
        if transforms is None:
            return
        for transform in transforms:
            trans, rot = conversions.hmat_to_trans_rot(transform['tfm'])
            self.transforms[transform['parent'],transform['child']] = (trans,rot)
        self.ready = True


def run_calibration_sequence ():
    
    rospy.init_node('calibration')
    
    yellowprint("Beginning calibration sequence.")
    
    tfm_pub = transform_publisher()
    tfm_pub.start()
    
    NUM_CAMERAS = 1
    cameras = cyni_cameras(NUM_CAMERAS)
    cameras.initialize_cameras()
        
    greenprint("Step 1. Calibrating mutliple cameras.")
    CAM_N_OBS = 10
    CAM_N_AVG = 5
    cam_calib = camera_calibrator(cameras)

    done = False
    while not done:
        cam_calib.calibrate(CAM_N_OBS, CAM_N_AVG)
        if not cam_calib.calibrated:
            redprint("Camera calibration failed.")
            cam_calib.reset_calibration()
        else:
            tfm_pub.add_transforms(cam_calib.get_transforms())
            if yes_or_no("Are you happy with the calibration? Check RVIZ."):
                done = True
            else:
                yellowprint("Calibrating cameras again.")
                cam_calib.reset_calibration()

    greenprint("Mutliple cameras calibrated.")

    greenprint("Step 2. Calibrating hydra and kinect.")
    HYDRA_N_OBS = 10
    HYDRA_N_AVG = 5
    HYDRA_AR_MARKER = 0

    hydra_calib = hydra_calibrator(cameras, ar_marker = HYDRA_AR_MARKER)

    done = False
    while not done:
        hydra_calib.calibrate(HYDRA_N_OBS, HYDRA_N_AVG)
        if not hydra_calib.calibrated:
            redprint("Camera calibration failed.")
            hydra_calib.reset_calibration()
        else:
            tfm_pub.add_transforms(hydra_calib.get_transforms())
            if yes_or_no("Are you happy with the calibration? Check RVIZ."):
                done = True
            else:
                yellowprint("Calibrating hydras.")
                hydra_calib.reset_calibration()

    greenprint("Hydra base calibrated.")

    greenprint("Step 3. Calibrating relative transforms of markers on gripper.")

    GRIPPER_MIN_OBS = 5
    GRIPPER_N_AVG = 5

    greenprint("Step 3.1 Calibrate l gripper.")
    l_gripper_calib = gripper_calibrator(cameras)
    
    # create calib_info based on gripper here
    calib_info = {'master':{'ar_markers':[0,1],
                            'hydras':['left'],
                            'angle_scale':0,
                            'master_group':1},
                  'l': {'ar_markers':[2,3,4],
                            'angle_scale':1},
                  'r': {'ar_markers':[5,6,7],
                            'angle_scale':-1}}
    
    l_gripper_calib.update_calib_info(calib_info)
    
    done = False
    while not done:
        l_gripper_calib.calibrate(GRIPPER_MIN_OBS, GRIPPER_N_AVG)
        if not l_gripper_calib.calibrated:
            redprint("Gripper calibration failed.")
            l_gripper_calib.reset_calibration()
        if yes_or_no("Are you happy with the calibration?"):
            done = True
        else:
            yellowprint("Calibrating l gripper again.")
            l_gripper_calib.reset_calibration()
            
    greenprint("Done with l gripper calibration.")

    greenprint("Step 3.2 Calibrate r gripper.")
    r_gripper_calib = gripper_calibrator(cameras)
    
    # create calib_info based on gripper here
    calib_info = {'master':{'ar_markers':[0,1],
                            'hydras':['left'],
                            'angle_scale':0,
                            'master_group':1},
                  'l': {'ar_markers':[2,3,4],
                            'angle_scale':1},
                  'r': {'ar_markers':[5,6,7],
                            'angle_scale':-1}}
    
    r_gripper_calib.update_calib_info(calib_info)
    
    done = False
    while not done:
        r_gripper_calib.calibrate(GRIPPER_MIN_OBS, GRIPPER_N_AVG)
        if not r_gripper_calib.calibrated:
            redprint("Gripper calibration failed.")
            r_gripper_calib.reset_calibration()
        if yes_or_no("Are you happy with the calibration?"):
            done = True
        else:
            yellowprint("Calibrating r gripper again.")
            r_gripper_calib.reset_calibration()

    greenprint("Done with r gripper calibration.")
    
    greenprint("Done with all the calibration.")
    
    while True:
        # stall
        time.sleep(0.1)