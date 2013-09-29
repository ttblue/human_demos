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


parser = argparse.ArgumentParser()
parser.add_argument('--name', help="Name of log file", required = True, type = str)
parser.add_argument('--n_tfm', help="Number of transforms to log", required = True, type=int)
args = parser.parse_args()

rospy.init_node('log_kinect_hydra_pr2')
time.sleep(10)

base_frame = 'base_footprint'
head_frame = 'head_plate_frame'
gripper_frame = 'l_gripper_tool_frame'
hydra_frame = 'hydra_base'
hydra_sensor = 'hydra_left'

tool_tfms = []
ar_tfms = []
head_tfms = []
hydra_tfms = []

tool_times = []
ar_times = []
hydra_times = []

i = 0

listener = tf.TransformListener()
time.sleep(3)
ar_markers = cameras.ARMarkersRos('/camera1_')

start = time.time()

while(i < args.n_tfm):
    #ar
    kinect_tfm, ar_time = ar_markers.get_marker_transforms(markers=[11], time_thresh=0.5, get_time=True)
    if kinect_tfm == {}:
        print "Lost sight of AR marker..."
        ar_tfms.append(None)
        ar_times.append(ar_time)
    else:
        ar_tfms.append(kinect_tfm[11])
        ar_times.append(ar_time)

    #pr2 gripper
    tool_time = listener.getLatestCommonTime(gripper_frame, base_frame)
    tool_trans, tool_quat = listener.lookupTransform(base_frame, gripper_frame, tool_time)

    #pr2 head
    head_trans, head_quat = listener.lookupTransform(base_frame, head_frame, rospy.Time(0))

    #hydra
    hydra_time = listener.getLatestCommonTime(hydra_sensor, base_frame)
    hydra_trans, hydra_quat = listener.lookupTransform(base_frame, hydra_sensor, hydra_time)

    tool_tfm = conversions.trans_rot_to_hmat(tool_trans, tool_quat)
    head_tfm = conversions.trans_rot_to_hmat(head_trans, head_quat)
    hydra_tfm = conversions.trans_rot_to_hmat(hydra_trans, hydra_quat)

    tool_tfms.append(tool_tfm)
    head_tfms.append(head_tfm)
    hydra_tfms.append(hydra_tfm)

    tool_times.append(tool_time)
    hydra_times.append(hydra_time)
    
    print i
    i = i+1
    time.sleep(1.0/30.5)

end = time.time()

ar_in_base_tfms = []
for i in xrange(len(ar_tfms)):
    if ar_tfms[i] == None:
        ar_in_base_tfms.append(None)
    else:
        head_frame_ar = T_h_k.dot(ar_tfms[i])
        base_frame_ar = head_tfms[i].dot(head_frame_ar)
        ar_in_base_tfms.append(base_frame_ar)

dic = {}
dic['ar_tfms'] = ar_in_base_tfms
dic['ar_times'] = ar_times
dic['tool_tfms'] = tool_tfms
dic['tool_times'] = tool_times
dic['hydra_tfms'] = hydra_tfms
dic['hydra_times'] = hydra_times
cPickle.dump( dic, open( args.name, "wa" ) )
print len(dic['ar_tfms'])
print len(dic['tool_tfms'])
print len(dic['hydra_tfms'])
print 'freq %f'%(len(tool_tfms)/(end-start))
print ''
#for i in xrange(len(ar_tfms)):
#    print dic['ar'][i]
#    print dic['tool'][i]

#except KeyboardInterrupt:
#    print "got Control-C"
