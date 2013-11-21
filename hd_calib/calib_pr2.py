#!/usr/bin/env python
import time
import os.path as osp
import argparse
import numpy as np, numpy.linalg as nlg
import cPickle

import rospy
import roslib; roslib.load_manifest('tf')
import tf;

from hd_utils import conversions, utils
from hd_utils.colorize import *
from hd_utils.yes_or_no import yes_or_no
from hd_utils.defaults import tfm_link_rof

from calibration_pipeline import CalibratedTransformPublisher
from cameras import RosCameras
from camera_calibration import CameraCalibrator
import gripper_calibration, gripper

tfl = None

def get_robot_kinect_transform():
    global tfl

    i = 5
    T_b_k = None
    while i > 0:
        try:
            
            now = rospy.Time.now()
            tfl.waitForTransform('base_footprint', 'camera_link', now, rospy.Duration(5.0))
            (trans, rot) = tfl.lookupTransform('base_footprint', 'camera_link', now)
            T_b_k = conversions.trans_rot_to_hmat(trans, rot)
            print T_b_k
            yellowprint("Got the transform for PR2")
            break
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            yellowprint("Failed attempt.")
            i -= 1
    return T_b_k

tfm_pub = None
cameras = None

CAM_N_OBS = 10
CAM_N_AVG = 50


def calibrate_cameras ():
    global tfm_pub, cameras, tfl
    
    Tbk = get_robot_kinect_transform()
    if yes_or_no('Calibrate again?'):
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
                print 1
                transform = cam_calib.get_transforms()[0]
                transform['child'] = 'camera_link'
    
                transform2 = {}
                transform2['parent'] = transform['parent']
                transform2['child']='base_footprint'
                transform2['tfm']=transform['tfm'].dot(nlg.inv(Tbk))
                tfm_pub.add_transforms([transform,transform2])
                if yes_or_no("Are you happy with the calibration? Check RVIZ."):
                    done = True
                else:
                    yellowprint("Calibrating cameras again.")
                    cam_calib.reset_calibration()
    
        greenprint("Cameras calibrated.")
    else:
        tfm_pub.load_calibration('cam_pr2')

    
    if yes_or_no('Get PR2 Gripper?'):
        # Finding transform between PR2's gripper and the tool_tip
        n_avg = 100
        mtfms = []
        while n_avg > 0:
            mtfms.append(cameras.get_ar_markers([1], camera=1)[1])
            time.sleep(0.03)
            n_avg -= 1
        mtfm = utils.avg_transform(mtfms)
    
        calib_file = 'calib_cam_hydra_gripper1'
        file_name = osp.join('/home/sibi/sandbox/human_demos/hd_data/calib',calib_file)
        with open(file_name,'r') as fh: calib_data = cPickle.load(fh)
    
        lr,graph = calib_data['grippers'].items()[0]
        gr = gripper.Gripper(lr, graph)
        assert 'tool_tip' in gr.mmarkers
        gr.tt_calculated = True
        
        t1m = gr.get_rel_transform(1, 'tool_tip', 0)
    
        i = 10
        Tbg = None
        while i > 0:
            try:
                now = rospy.Time.now()
                tfl.waitForTransform('base_footprint','l_gripper_tool_frame', now, rospy.Duration(5.0))
                (trans, rot) = tfl.lookupTransform('base_footprint','l_gripper_tool_frame',rospy.Time(0))
                Tbg = conversions.trans_rot_to_hmat(trans, rot)
                break
            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                yellowprint("Failed attempt.")            
                i -= 1
                pass
        if Tbg is not None:
            tga = {'parent':'l_gripper_tool_frame',
                   'child':'pr2_lgripper_tooltip',
                    'tfm':nlg.inv(Tbg).dot(Tbk).dot(tfm_link_rof).dot(mtfm).dot(t1m)}
            tfm_pub.add_transforms([tga])
            greenprint("Gripper to marker found.")
        else:
            redprint("Gripper to marker not found.")


def initialize_calibration():
    global tfm_pub, cameras, tfl
    rospy.init_node('calib_pr2')
    tfl = tf.TransformListener()
    cameras = RosCameras(2)
    tfm_pub = CalibratedTransformPublisher()#cameras)
    tfm_pub.start()


if __name__ == '__main__':
    initialize_calibration()
    calibrate_cameras()
     
