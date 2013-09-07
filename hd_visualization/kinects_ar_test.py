#!/usr/bin/env python
import cv2
import numpy as np
import subprocess
import cyni
import Image
import time
import rospy
import roslib
roslib.load_manifest('tf')
import tf
roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse
import argparse
import cPickle
from hd_utils import ros_utils as ru, clouds, conversions
primesense_carmine_f = 544.260779961



getMarkers = None
req = MarkerPositionsRequest()


def avg_quaternions(qs):
    """
    Returns the "average" quaternion of the quaternions in the list qs.
    ref: http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
    """
    M = np.zeros((4,4))
    for q in qs:
        q = q.reshape((4,1))
        M = M + q.dot(q.T)

    l, V = np.linalg.eig(M)
    q_avg =  V[:, np.argmax(l)]
    return q_avg/np.linalg.norm(q_avg)


def get_ar_transform_id (depth, rgb, idm=None):
    """
    In order to run this, ar_marker_service needs to be running.
    """
    req.pc = ru.xyzrgb2pc(clouds.depth_to_xyz(depth, primesense_carmine_f), rgb, '/camera_link')
    res = getMarkers(req)
    marker_tfm = {marker.id:conversions.pose_to_hmat(marker.pose.pose) for marker in res.markers.markers}

    if not idm: return marker_tfm
    if idm not in marker_tfm: return None
    return marker_tfm[idm]



global getMarkers


cmap = np.zeros((256, 3),dtype='uint8')
cmap[:,0] = range(256)
cmap[:,2] = range(256)[::-1]
cmap[0] = [0,0,0]

#grabber = cloudprocpy.CloudGrabber("#1")
#grabber.startRGBD()
#g1 = cloudprocpy.CloudGrabber("#1")
#g1.startRGBD()

rospy.init_node('kinects_ar_test')
getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_tfm', required=True, type=int)
args = parser.parse_args()


cyni.initialize()

device_list = cyni.enumerateDevices()
d1 = cyni.Device(device_list[0]['uri'])
d2 = cyni.Device(device_list[1]['uri'])
d1.open()
d2.open()
k1_ts = np.empty((0, 3))
k1_qs = np.empty((0,4))
k2_ts = np.empty((0, 3))
k2_qs = np.empty((0,4))

#subprocess.call("sudo killall XnSensorServer", shell=True)
ds1 = d1.createStream("depth", width=640, height = 480, fps=30)
ds1.start()
cs1 = d1.createStream("color", width=640, height = 480, fps = 30)
cs1.start()
ds2 = d2.createStream("depth", width=640, height = 480, fps = 30)
ds2.start()
cs2 = d1.createStream("color", width=640, height = 480, fps = 30)
cs2.start()
i = 0
while(i < args.n_tfm):
    tfm1 = None
    tfm2 = None 
    rgb1 = cs1.readFrame()
    #cv2.imshow("rgb", rgb.data)
    depth1 = ds1.readFrame()
    #cv2.imshow("depth", cmap[np.fmin((depth.data*.064).astype('int'), 255)])
    tfm1 = get_ar_transform_id(depth1.data, rgb1.data, 2)
    print tfm1
    if tfm1 == None:
        print "got none"
        continue
    #print tfm
    #tfms1.append(tfm1)

    k1_t, k1_q = conversions.hmat_to_trans_rot(tfm1)
    k1_ts = np.r_[k1_ts, np.array(k1_t, ndmin=2)]
    k1_qs = np.r_[k1_qs, np.array(k1_q, ndmin=2)]

    rgb2 = cs2.readFrame()
    depth2 = ds2.readFrame()
    tfm2 = get_ar_transform_id(depth2.data, rgb2.data, 2)
    #print tfm 
    #trans, rot = conversions.hmat_to_trans_rot(tfm)
    print tfm2
    if tfm2 == None:
        print "got none"
        continue
    k2_t, k2_q = conversions.hmat_to_trans_rot(tfm2)
    k2_ts = np.r_[k2_ts, np.array(k2_t, ndmin=2)]
    k2_qs = np.r_[k2_qs, np.array(k2_q, ndmin=2)]
    print i
    i = i+1
    #except:
    #    print "Warning: Problem with ar"
    #    pass
    #cv2.waitKey(30)

k1_trans_avg = np.sum(k1_ts, axis=0) / args.n_tfm
k1_rot_avg = avg_quaternions(k1_qs)
k1_tfm = conversions.trans_rot_to_hmat(k1_trans_avg, k1_rot_avg)
print k1_tfm

#except KeyboardInterrupt:
#    print "got Control-C"
