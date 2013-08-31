#!/usr/bin/ipython -i

import rospy
from sensor_msgs.msg import PointCloud2

import cloudprocpy as cpr

from hd_utils import ros_utils as ru, clouds

asus_xtion_pro_f = 544.260779961

def visualize_pointcloud (camera_frame="camera_depth_optical_frame", device_id="#1"):
    """
    Visualize point clouds from openni grabber through ROS.
    """
    if rospy.get_name() == '/unnamed':
        rospy.init_node("visualize_pointcloud")
    
    
    pc_pub = rospy.Publisher("camera_points", PointCloud2)
    sleeper = rospy.Rate(30)
    
    grabber = cpr.CloudGrabber(device_id)
    grabber.startRGBD()
    
    print "Streaming now from frame %s: Pointclouds only."%camera_frame
    while True:
        try:
            r, d = grabber.getRGBD()            

            pc = ru.xyzrgb2pc(clouds.depth_to_xyz(d, asus_xtion_pro_f), r, camera_frame)
            pc_pub.publish(pc)

            sleeper.sleep()
        except KeyboardInterrupt:
            print "Keyboard interrupt. Exiting."
            break
