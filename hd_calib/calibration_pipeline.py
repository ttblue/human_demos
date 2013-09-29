#!/usr/bin/python
import numpy as np
from threading import Thread
import time
import cPickle

import roslib, rospy
roslib.load_manifest('tf')
import tf
from sensor_msgs.msg import PointCloud2

from hd_utils.colorize import *
from hd_utils.yes_or_no import yes_or_no
from hd_utils import conversions, clouds, ros_utils as ru

from cameras import RosCameras
from camera_calibration import CameraCalibrator
from hydra_calibration import HydraCalibrator
from gripper_calibration import GripperCalibrator
import get_marker_transforms as gmt

np.set_printoptions(precision=5, suppress=True)
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

class CalibratedTransformPublisher(Thread):

    def __init__(self, cameras=None):
        Thread.__init__(self)
        
        if rospy.get_name() == '/unnamed':
            rospy.init_node('calibration_node')

        self.transforms = {}
        self.ready = False
        self.rate = 30.0
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.cameras = cameras
        self.publish_grippers = False
        self.grippers = {}

    def run (self):
        """
        Publishes the transforms stored.
        """
        while True:
            if self.ready:
                for parent, child in self.transforms:
                    trans, rot = self.transforms[parent, child]
                    self.tf_broadcaster.sendTransform(trans, rot,
                                                      rospy.Time.now(),
                                                      child, parent)
                if self.publish_grippers:
                    self.publish_gripper_tfms()
            time.sleep(1/self.rate)

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
        
    def add_gripper (self, gripper):
        self.grippers[gripper.lr] = gripper

    def publish_gripper_tfms (self):
        marker_tfms = self.cameras.get_ar_markers()
        theta = gmt.get_pot_angle()
        parent_frame = self.cameras.parent_frame

        transforms = []
        for gripper in self.grippers.values():
            transforms += gripper.get_all_transforms(marker_tfms, theta, parent_frame)
            
        for transform in transforms:
            trans, rot = conversions.hmat_to_trans_rot(transform['tfm'])
            self.tf_broadcaster.sendTransform(trans, rot,
                                              rospy.Time.now(),
                                              transform['child'],
                                              transform['parent'])

    def set_publish_grippers(self, val=None):
        """
        Toggles by default
        """
        if val is None:
            self.publish_grippers = not self.publish_grippers
        else: self.publish_grippers = not not val
        
        if self.publish_grippers: self.ready = True


    def reset (self):
        self.ready = False
        self.publish_grippers = False
        self.transforms = {}
        self.grippers = {}
        
    def load_calibration(self, file):
        """
        Use this if experimental setup has not changed.
        Load files which have been saved by this class. Specific format involved.
        """
        self.reset()
        
        with open(file,'r') as fh: calib_data = cPickle.load(file)
        for gripper in calib_data['grippers']:
            self.add_gripper(gripper)    
        self.add_transforms(calib_data['transforms'])
        
    def save_calibration(self, file):
        """
        Save the transforms and the gripper data from this current calibration.
        This assumes that the entire calibration data is stored in this class.
        """
        
        calib_data = {}
        calib_data['grippers'] = self.grippers.values()
        calib_transforms = []
        for parent, child in self.transforms:
            tfm = {}
            tfm['parent'] = parent
            tfm['child'] = child
            trans, rot = self.transforms[parent, child]
            tfm['tfm'] = conversions.trans_rot_to_hmat(trans, rot)
            calib_transforms.append(tfm)
        
        calib_data['transforms'] = calib_transforms
        with open(file, 'w') as fh: cPickle.dump(calib_data, fh)

#Global variables
cameras = None
tfm_pub = None

CAM_N_OBS = 1
CAM_N_AVG = 5


def calibrate_cameras ():
    global cameras, tfm_pub
    
    greenprint("Step 1. Calibrating mutliple cameras.")
    cam_calib = CameraCalibrator(cameras)

    done = False
    while not done:
        cam_calib.calibrate(CAM_N_OBS, CAM_N_AVG)
        if cameras.num_cameras == 1:
            break
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

    greenprint("Cameras calibrated.")

HYDRA_N_OBS = 10
HYDRA_N_AVG = 50
CALIB_CAMERA = 0

def calibrate_hydras ():
    global cameras, tfm_pub
    
    greenprint("Step 2. Calibrating hydra and kinect.")
    HYDRA_AR_MARKER = input('Enter the ar_marker you are using for calibration.')
    hydra_calib = HydraCalibrator(cameras, ar_marker = HYDRA_AR_MARKER, calib_camera=CALIB_CAMERA)

    done = False
    while not done:
        hydra_calib.calibrate('camera', HYDRA_N_OBS, HYDRA_N_AVG)
        if not hydra_calib.calibrated:
            redprint("Hydra calibration failed.")
            hydra_calib.reset_calibration()
        else:
            tfm_pub.add_transforms(hydra_calib.get_transforms('camera'))
            if yes_or_no("Are you happy with the calibration? Check RVIZ."):
                done = True
            else:
                yellowprint("Calibrating hydras.")
                hydra_calib.reset_calibration()

    greenprint("Hydra base calibrated.")


GRIPPER_MIN_OBS = 4
GRIPPER_N_AVG = 20
l_gripper_calib = None

def calibrate_grippers ():
    global cameras, tfm_pub, l_gripper_calib

    greenprint("Step 3. Calibrating relative transforms of markers on gripper.")

    greenprint("Step 3.1 Calibrate l gripper.")
    l_gripper_calib = GripperCalibrator(cameras, 'l')
    
    # create calib_info based on gripper here
    calib_info = {'master':{'ar_markers':[1],#,3,10,13],
                            #'hydras':['left'],
                            'angle_scale':0,
                            'master_marker':1},
                  'l': {'ar_markers':[15],#,11],
                            'angle_scale':1},
                  'r': {'ar_markers':[4],#,6],
                            'angle_scale':-1}}
    
    l_gripper_calib.update_calib_info(calib_info)
    
    done = False
    while not done:
        l_gripper_calib.calibrate(GRIPPER_MIN_OBS, GRIPPER_N_AVG)
        if not l_gripper_calib.calibrated:
            redprint("Gripper calibration failed.")
            l_gripper_calib.reset_calibration()
        
        tfm_pub.add_gripper(l_gripper_calib.get_gripper())
        tfm_pub.set_publish_grippers(True)

        if yes_or_no("Are you happy with the calibration?"):
            done = True
        else:
            yellowprint("Calibrating l gripper again.")
            l_gripper_calib.reset_calibration()
            tfm_pub.set_publish_grippers(False)
    

    greenprint("Done with l gripper calibration.")

#     greenprint("Step 3.2 Calibrate r gripper.")
#     r_gripper_calib = GripperCalibrator(cameras, 'r')
#     
#     # create calib_info based on gripper here
#     calib_info = {'master':{'ar_markers':[0,1],
#                             'hydras':['left'],
#                             'angle_scale':0,
#                             'master_group':1},
#                   'l': {'ar_markers':[2,3,4],
#                             'angle_scale':1},
#                   'r': {'ar_markers':[5,6,7],
#                             'angle_scale':-1}}
#     
#     r_gripper_calib.update_calib_info(calib_info)
#     
#     done = False
#     while not done:
#         r_gripper_calib.calibrate(GRIPPER_MIN_OBS, GRIPPER_N_AVG)
#         if not r_gripper_calib.calibrated:
#             redprint("Gripper calibration failed.")
#             r_gripper_calib.reset_calibration()
#         if yes_or_no("Are you happy with the calibration?"):
#             done = True
#         else:
#             yellowprint("Calibrating r gripper again.")
#             r_gripper_calib.reset_calibration()
#     
#     tfm_pub.add_gripper(r_gripper_calib.get_gripper())
#                         
#     greenprint("Done with r gripper calibration.")


NUM_CAMERAS = 1
def initialize_calibration():
    global cameras, tfm_pub
    rospy.init_node('calibration')
    cameras = RosCameras(num_cameras=NUM_CAMERAS)
    tfm_pub = CalibratedTransformPublisher(cameras)
    tfm_pub.start()




def run_calibration_sequence (spin=False):
        
    yellowprint("Beginning calibration sequence.")
    calibrate_cameras()
    calibrate_hydras() 
    calibrate_grippers ()

    greenprint("Done with all the calibration.")
    
    while True and spin:
        # stall
        time.sleep(0.1)