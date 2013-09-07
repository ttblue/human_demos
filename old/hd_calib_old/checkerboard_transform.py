#!/usr/bin/ipython -i
import cv
import time
import numpy as np

import rospy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped


import cloudprocpy as cpr
from hd_utils import clouds, conversions, ros_utils

cb_rows = 8
cb_cols = 6

asus_xtion_pro_f = 544.260779961

def print_checker_board_transform ():
    pass

def get_corners_rgb(rgb,rows=None,cols=None):
    cv_rgb = cv.fromarray(rgb)
    
    if not rows: rows = cb_rows
    if not cols: cols = cb_cols
    
    rtn, corners = rtn, corners = cv.FindChessboardCorners(cv_rgb, (cb_rows, cb_cols))
    return rtn, corners

def get_xyz_from_corners (corners, xyz):
    points = []
    for j,i in corners:
        points.append(xyz[round(i),round(j)])
    return np.asarray(points)

def get_svd_tfm_from_points (points):
    points = np.asarray(points)
    avg = points.sum(axis=0)/points.shape[0]
    points = points- avg
    u,s,v = np.linalg.svd(points, full_matrices=True)
    #nm = v[2,:].T
    rot = v.T
    tfm = np.eye(4)
    tfm[0:3,0:3] = rot
    tfm[0:3,3] = avg
    return tfm#nm

def checkerboard_loop ():
    
    rospy.init_node("checkerboard_tracker")
    
    grabber = cpr.CloudGrabber()
    grabber.startRGBD()
    
    pc_pub = rospy.Publisher("cb_cloud", PointCloud2)
    pose_pub = rospy.Publisher("cb_pose", PoseStamped)
    
    rate = rospy.Rate(10)
    
    while True:
        try:
            rgb, depth = grabber.getRGBD()
            xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
            
            rtn, corners = get_corners_rgb (rgb)
            points = get_xyz_from_corners(corners, xyz)
            
            print "Shape of points: ", points.shape
            
            if points.any():
                try:
                    tfm = get_svd_tfm_from_points(points)
                except:
                    continue
                print tfm
                
                pc = ros_utils.xyzrgb2pc(xyz, rgb, "/map")
                pose = PoseStamped()
                pose.pose = conversions.hmat_to_pose(tfm)
                
                pc.header.frame_id = pose.header.frame_id = "/map"
                pose.header.stamp = pose.header.stamp = rospy.Time.now()
                
                pose_pub.publish(pose)
                pc_pub.publish(pc)
                
                rate.sleep()
            
        except KeyboardInterrupt:
            break