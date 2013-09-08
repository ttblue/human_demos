#!/usr/bin/env python
import cv2
import numpy as np, numpy.linalg as nlg
import subprocess
import cyni
import Image
import time
import rospy
import roslib
roslib.load_manifest('tf')
import tf
import argparse
import cPickle
from hd_utils import ros_utils as ru, clouds, conversions
from hd_calib import cameras
primesense_carmine_f = 544.260779961

T_h_k = np.array([[-0.02102462, -0.03347223,  0.99921848, -0.186996  ],
 [-0.99974787, -0.00717795, -0.02127621,  0.04361884],
 [ 0.0078845,  -0.99941387, -0.03331288,  0.22145804],
 [ 0.,          0.,          0.,          1.        ]])


cmap = np.zeros((256, 3),dtype='uint8')
cmap[:,0] = range(256)
cmap[:,2] = range(256)[::-1]
cmap[0] = [0,0,0]

parser = argparse.ArgumentParser()
parser.add_argument('--name', help="Name of log file", required = True, type = str)
parser.add_argument('--n_tfm', help="Number of transforms to log", required = True, type=int)
args = parser.parse_args()

rospy.init_node('log_kinect_hydra_pr2')


base_frame = 'base_footprint'
head_frame = 'head_plate_frame'
gripper_frame = 'r_gripper_tool_frame'
hydra_frame = 'hydra_base'
hydra_sensor = 'hydra_left'
tool_tfms = []
ar_tfms = []
head_tfms = []
hydra_tfms = []
i = 0

listener = tf.TransformListener()
time.sleep(3)
ar_markers = ar_markers_ros('/camera_')

start = time.time()
while(i < args.n_tfm):
        
    kinect_tfm = ar_markers.get_marker_transforms(markers=[13], time_thresh=0)
        if kinect_tfm == {}:
            print "Lost sight of AR marker. Breaking..."
            continue
    tool_trans, tool_quat = listener.lookupTransform(base_frame, gripper_frame, rospy.Time(0))
    head_trans, head_quat = listener.lookupTransform(base_frame, head_frame, rospy.Time(0))
    hydra_trans, hydra_quat = listener.lookupTransform(base_frame, hydra_sensor, rospy.Time(0))
    tool_tfm = conversions.trans_rot_to_hmat(tool_trans, tool_quat)
    head_tfm = conversions.trans_rot_to_hmat(head_trans, head_quat)
    hydra_tfm = conversions.trans_rot_to_hmat(hydra_trans, hydra_quat)
    tool_tfms.append(tool_tfm)
    head_tfms.append(head_tfm)
    ar_tfms.append(kinect_tfm[13])
    hydra_tfms.append(hydra_tfm)
    #trans, rot = conversions.hmat_to_trans_rot(tfm)
    print i
    i = i+1
end = time.time()

ar_in_base_tfms = []
for i in xrange(len(ar_tfms)):
    head_frame_ar = T_h_k.dot(ar_tfms[i])
    base_frame_ar = head_tfms[i].dot(head_frame_ar)
    ar_in_base_tfms.append(base_frame_ar)

dic = {}
dic['kinect'] = ar_in_base_tfms
dic['pr2'] = tool_tfms
dic['hydra'] = hydra_tfms
cPickle.dump( dic, open( args.name, "wa" ) )
print len(dic['kinect'])
print len(dic['pr2'])
print len(dic['hydra'])
print 'freq %f'%(len(tool_tfms)/(end-start))
print ''
#for i in xrange(len(ar_tfms)):
#    print dic['ar'][i]
#    print dic['tool'][i]

#except KeyboardInterrupt:
#    print "got Control-C"
