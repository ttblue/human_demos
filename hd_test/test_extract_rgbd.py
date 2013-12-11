import numpy as np, numpy.linalg as nlg
import rospy
import os, os.path as osp
import cPickle
import cv2

import rosbag
import roslib
roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse
from ar_track_alvar.msg import AlvarMarkers

getMarkers = None
req = MarkerPositionsRequest()

from hd_utils.defaults import tfm_link_rof, asus_xtion_pro_f
from hd_utils.colorize import *
from hd_utils import ros_utils as ru, clouds, conversions, extraction_utils as eu


def get_ar_marker_poses (pc):
    global getMarkers, req
    
    if rospy.get_name() == '/unnamed':
        rospy.init_node('keypoints')
    
    if getMarkers is None:
        getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)
    
    req.pc = pc
    
    marker_tfm = {}
    res = getMarkers(req)
    for marker in res.markers.markers:
        marker_tfm[marker.id] = conversions.pose_to_hmat(marker.pose.pose).tolist()
    
    #print "Marker ids found: ", marker_tfm.keys()
    
    return marker_tfm



def get_tfm_of_camera(ind1,ind2):

    rgbd1_dir = '/home/sibi/sandbox/human_demos/hd_data/demos/sibi_demo/camera_#1'
    rgbd2_dir = '/home/sibi/sandbox/human_demos/hd_data/demos/sibi_demo/camera_#2'
    rgbs1fnames, depths1fnames, stamps1 = eu.get_rgbd_names_times(rgbd1_dir)
    rgbs2fnames, depths2fnames, stamps2 = eu.get_rgbd_names_times(rgbd2_dir)

    winname = 'abcd'

    ar1, ar2 = None, None
    rgb1 = cv2.imread(rgbs1fnames[ind1])
    assert rgb1 is not None
    depth1 = cv2.imread(depths1fnames[ind1],2)
    assert depth1 is not None
    
    cv2.imshow(winname,)
    
    xyz1 = clouds.depth_to_xyz(depth1, asus_xtion_pro_f)
    pc1 = ru.xyzrgb2pc(xyz1, rgb1, frame_id='', use_time_now=False)
    ar_tfms1 = get_ar_marker_poses(pc1)
    if ar_tfms1 and 1 in ar_tfms1:
        blueprint("Got markers " + str(ar_tfms1.keys()) + " at time %f"%stamps1[ind1])
        ar1 = ar_tfms1[1]
        
    rgb2 = cv2.imread(rgbs2fnames[ind2])
    assert rgb2 is not None
    depth2 = cv2.imread(depths2fnames[ind2],2)
    assert depth2 is not None
    xyz2 = clouds.depth_to_xyz(depth2, asus_xtion_pro_f)
    pc2 = ru.xyzrgb2pc(xyz2, rgb2, frame_id='', use_time_now=False)
    ar_tfms2 = get_ar_marker_poses(pc2)
    if ar_tfms2 and 1 in ar_tfms2:
        blueprint("Got markers " + str(ar_tfms2.keys()) + " at time %f"%stamps1[ind2])
        ar2 = ar_tfms2[1]

    if ar1 is None or ar2 is None:
        return None
    return ar1.dot(nlg.inv(ar2))