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
from hd_utils.defaults import asus_xtion_pro_f



getMarkers = None
req = MarkerPositionsRequest()

def get_ar_transform_id (depth, rgb, idm=None):
    """
    In order to run this, ar_marker_service needs to be running.
    """
    req.pc = ru.xyzrgb2pc(clouds.depth_to_xyz(depth, asus_xtion_pro_f), rgb, '/camera_link')
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


cyni.initialize()

device = cyni.getAnyDevice()
#subprocess.call("sudo killall XnSensorServer", shell=True)
device.open()
colorStream = device.createStream("color", width=640, height=480, fps=30)
colorStream.start()
depthStream = device.createStream("depth", width=640, height = 480, fps=30)
depthStream.start()
tfms = []
start = time.time()
i = 0
while(i < args.n_tfm):
        
    rgb = colorStream.readFrame()
    #cv2.imshow("rgb", rgb.data)
    depth = depthStream.readFrame()
    #cv2.imshow("depth", cmap[np.fmin((depth.data*.064).astype('int'), 255)])
    tfm = get_ar_transform_id(depth.data, rgb.data, 2)
    print i
    print tfm 
    #trans, rot = conversions.hmat_to_trans_rot(tfm)
    if tfm == None:
        print "got none"
        continue
    tfms.append(tfm)
    i = i+1
    #except:
    #    print "Warning: Problem with ar"
    #    pass
    #cv2.waitKey(30)

print len(tfms)
print time.time() - start
cPickle.dump( tfms, open( args.name, "wa" ) )


#except KeyboardInterrupt:
#    print "got Control-C"
