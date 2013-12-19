## simple script to save a guess transform
## and two point-clouds from 2 depth-cameras.

import roslib, rospy, sys

roslib.load_manifest('tf')
import tf

import roslib; roslib.load_manifest('pcd_service')
from pcd_service.srv import PCDData, PCDDataRequest, PCDDataResponse

import numpy as np

from hd_utils import ros_utils as ru, clouds, conversions, utils
from hd_utils.colorize import *
from hd_calib.cameras import RosCameras

'''
Captures cloud such that IR interference is avoided.
Makes the user cover cameras while taking data from the
other camera.
'''
def save_clouds():
    NUM_CAM      = 2   
    cameras      = RosCameras(NUM_CAM)
    tfm_listener = tf.TransformListener()
    fr1 = camera1_rgb_optical_frame
    fr2 = camera2_rgb_optical_frame
    c1, c2 = 0, 1

    print "Waiting for service .."
    rospy.wait_for_service('pcd_service')
    pcdService = rospy.ServiceProxy("pcd_service", PCDData)
    
    raw_input(colorize("Do not move the objects on the table now.", "green", True))

    raw_input(colorize("Cover camera %i and hit enter!" % (c2 + 1), 'yellow', True))
    pc1 = cameras.get_pointcloud(c1)
    raw_input(colorize("Cover camera %i and hit enter!" % (c1 + 1), 'yellow', True))
    pc2 = cameras.get_pointcloud(c2)
    pc2 = ru.transformPointCloud2(pc2, tfm_listener, fr1, fr2)
    req = PCDDataRequest()

    req.pc = pc1
    req.fname = "cloud1.pcd"
    pcdService(req)

    req.pc = pc2
    req.fname = "cloud2.pcd"
    pcdService(req)



if __name__=='__main__':
    rospy.init_node('capture_clouds')
    save_clouds()
