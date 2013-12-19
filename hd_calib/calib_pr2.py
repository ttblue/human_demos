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
from hd_utils.defaults import tfm_link_rof, calib_files_dir

from calibration_pipeline import CalibratedTransformPublisher
from cameras import RosCameras
from camera_calibration import CameraCalibrator
import gripper_calibration, gripper

tf_listener = None

def get_robot_kinect_transform():
    global tf_listener

    tfm_base_kinect = None
    for i in xrange(5):
        try:
            now = rospy.Time.now()
            tf_listener.waitForTransform('base_footprint', 'camera_link', now, rospy.Duration(5.0))
            (trans, rot) = tf_listener.lookupTransform('base_footprint', 'camera_link', now)
            tfm_base_kinect = conversions.trans_rot_to_hmat(trans, rot)
            print tfm_base_kinect
            yellowprint("Got the transform for PR2")
            break
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            yellowprint("Failed attempt.")

    return tfm_base_kinect

tfm_pub = None
cameras = None

CAM_N_OBS = 10
CAM_N_AVG = 50


def calibrate_cameras ():
    global tfm_pub, cameras, tf_listener
    
    tfm_base_kinect = get_robot_kinect_transform()
    if yes_or_no('Calibrate again?'):
        greenprint("Step 1. Calibrating multiple cameras.")
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
                tfm_reference_camera = cam_calib.get_transforms()[0]
                tfm_reference_camera['child'] = 'camera_link'
    
                tfm_base_reference = {}
                tfm_base_reference['child'] = transform['parent']
                tfm_base_reference['parent'] = 'base_footprint'
                tfm_base_reference['tfm'] = nlg.inv(tfm_reference_camera['tfm'].dot(nlg.inv(tfm_base_kinect)))
                tfm_pub.add_transforms([tfm_base_reference])
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
        marker_id = 1
        camera_id = 1
        n_avg = 100
        tfms_camera_marker = []
        for i in xrange(n_avg):
            tfms_camera_marker.append(cameras.get_ar_markers([marker_id], camera=camera_id)[marker_id])
            time.sleep(0.03)
        tfm_camera_marker = utils.avg_transform(tfms_camera_marker)
    
        calib_file = 'calib_cam_hydra_gripper1'
        file_name = osp.join(calib_files_dir, calib_file)
        with open(file_name, 'r') as fh: calib_data = cPickle.load(fh)
    
        lr, graph = calib_data['grippers'].items()[0]
        gr = gripper.Gripper(lr, graph)
        assert 'tool_tip' in gr.mmarkers
        gr.tt_calculated = True
        
        tfm_marker_tooltip = gr.get_rel_transform(marker_id, 'tool_tip', 0)
    
        i = 10
        tfm_base_gripper = None
        while i > 0:
            try:
                now = rospy.Time.now()
                tf_listener.waitForTransform('base_footprint', 'l_gripper_tool_frame', now, rospy.Duration(5.0))
                (trans, rot) = tf_listener.lookupTransform('base_footprint', 'l_gripper_tool_frame', rospy.Time(0))
                tfm_base_gripper = conversions.trans_rot_to_hmat(trans, rot)
                break
            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                yellowprint("Failed attempt.")            
                i -= 1
                pass
        if tfm_base_gripper is not None:
            tfm_gripper_tooltip = {'parent':'l_gripper_tool_frame',
                                   'child':'pr2_lgripper_tooltip',
                                   'tfm':nlg.inv(tfm_base_gripper).dot(tfm_base_kinect).dot(tfm_link_rof).dot(tfm_camera_marker).dot(tfm_marker_tooltip)
                                   }
            tfm_pub.add_transforms([tfm_gripper_tooltip])
            greenprint("Gripper to marker found.")
        else:
            redprint("Gripper to marker not found.")


def initialize_calibration():
    global tfm_pub, cameras, tf_listener
    rospy.init_node('calib_pr2')
    tf_listener = tf.TransformListener()
    cameras = RosCameras(2)
    tfm_pub = CalibratedTransformPublisher()  # cameras)
    tfm_pub.start()


if __name__ == '__main__':
    initialize_calibration()
    calibrate_cameras()
     
