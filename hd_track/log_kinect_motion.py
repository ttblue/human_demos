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
roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse
import argparse
import cPickle
from hd_utils import ros_utils as ru, clouds, conversions
primesense_carmine_f = 544.260779961

T_h_k = np.array([[-0.02102462, -0.03347223,  0.99921848, -0.186996  ],
 [-0.99974787, -0.00717795, -0.02127621,  0.04361884],
 [ 0.0078845,  -0.99941387, -0.03331288,  0.22145804],
 [ 0.,          0.,          0.,          1.        ]])

getMarkers = None
req = MarkerPositionsRequest()

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
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="Name of log file", required = True, type = str)
parser.add_argument('--n_tfm', help="Number of transforms to log", required = True, type=int)
args = parser.parse_args()

rospy.init_node('log_kinect_static')
getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)


base_frame = 'base_link'
head_frame = 'head_plate_frame'
gripper_frame = 'r_gripper_tool_frame'

cyni.initialize()
device = cyni.getAnyDevice()
#subprocess.call("sudo killall XnSensorServer", shell=True)
device.open()
colorStream = device.createStream("color", width=640, height=480, fps=30)
colorStream.start()
depthStream = device.createStream("depth", width=640, height = 480, fps=30)
depthStream.start()
ar_tfms = []
tool_tfms = []
head_tfms = []
i = 0

listener = tf.TransformListener()
time.sleep(3)
while(i < args.n_tfm):
        
    rgb = colorStream.readFrame()
    #cv2.imshow("rgb", rgb.data)
    depth = depthStream.readFrame()
    #cv2.imshow("depth", cmap[np.fmin((depth.data*.064).astype('int'), 255)])
    tool_trans, tool_quat = listener.lookupTransform(base_frame, gripper_frame, rospy.Time(0))
    head_trans, head_quat = listener.lookupTransform(base_frame, head_frame, rospy.Time(0))
    ar_tfm = get_ar_transform_id(depth.data, rgb.data, 2)
    if ar_tfm == None:
        print "got none"
        continue
    tool_tfm = conversions.trans_rot_to_hmat(tool_trans, tool_quat)
    head_tfm = conversions.trans_rot_to_hmat(head_trans, head_quat)
    #print head_tfm
    #print tool_tfm
    tool_tfms.append(tool_tfm)
    head_tfms.append(head_tfm)
    ar_tfms.append(ar_tfm)
    #trans, rot = conversions.hmat_to_trans_rot(tfm)
    print i
    i = i+1


ar_in_base_tfms = []
for i in xrange(len(ar_tfms)):
    head_frame_ar = T_h_k.dot(ar_tfms[i])
    base_frame_ar = head_tfms[i].dot(head_frame_ar)
    ar_in_base_tfms.append(base_frame_ar)

dic = {}
dic['ar'] = ar_in_base_tfms
dic['tool'] = tool_tfms
cPickle.dump( dic, open( args.name, "wa" ) )
print len(dic['ar'])
print len(dic['tool'])
print ''
#for i in xrange(len(ar_tfms)):
#    print dic['ar'][i]
#    print dic['tool'][i]

#except KeyboardInterrupt:
#    print "got Control-C"
