#!/usr/bin/python
import numpy as np
import cyni
from threading import Thread
import time

import roslib, rospy
roslib.load_manifest('tf')
import tf

from hd_utils.colorize import *
from hd_utils import conversions

from cyni_camera_calibration import camera_calibrator
"""
Steps to be taken:
    1.  Calibrate cameras w.r.t. each other.
        Publish transforms.
        ******** DONE
    2.  Calibrate hydras w.r.t. cameras.
        Publish transforms.
    3.  Calibrate potentiometer.
        Calibrate gripper + angle.
        Publish transform or make graph available to get transforms.
    
    Flesh it out:
    Camera calibration --- what do we need?
    1. Get data from cyni.
    
"""


class transform_publisher(Thread):

    def __init__(self):
        Thread.__init__(self)
        
        if rospy.get_name() == '/unnamed':
            rospy.init_node('calibration_node')
        
        self.transforms = []
        self.ready = False
        self.rate = 30.0
        self.tf_broadcaster = tf.TransformBroadcaster()
    
    def run (self):
        """
        Publishes the transforms stored.
        """
        while True:
            if self.ready:
                for trans, rot, parent, child in self.transforms:
                    self.tf_broadcaster.sendTransform(trans, rot,
                                                      rospy.Time.now(),
                                                      child, parent)
            time.sleep(1/self.rate)
    
    def add_transforms(self, transforms):
        """
        Takes a list of dicts with relevant transform information.
        """
        for transform in transforms:
            trans, rot = conversions.hmat_to_trans_rot(transform['tfm'])
            self.transforms.append([trans,rot,transform['parent'],transform['child']])
        self.ready = True
        

def run_calibration_sequence ():
    yellowprint("Beginning calibration sequence.")
    
    tfm_pub = transform_publisher()
    tfm_pub.start()
        
    greenprint("Step 1. Calibrating mutliple cameras.")
    CAM_CAM_N_OBS = 10
    CAM_CAM_N_AVG = 5
    cam_calib = camera_calibrator(num_cameras=2)

    done = False
    while not done:
        cam_calib.calibrate(CAM_CAM_N_OBS, CAM_CAM_N_AVG)
        if not cam_calib.calibrated:
            redprint("Camera calibration failed.")
            cam_calib.reset_calibration()
        else:
            done = True
    
    tfm_pub.add_transforms(cam_calib.get_transforms())