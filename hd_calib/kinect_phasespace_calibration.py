#!/usr/bin/ipython -i

import time
import numpy as np, numpy.linalg as nlg
import scipy as scp, scipy.linalg as sclg, scipy.optimize as sco
import roslib; roslib.load_manifest("tf")
import rospy, tf
import roslib; roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse


import cloudprocpy as cpr
import phasespace as ph

from hd_utils import clouds, ros_utils as ru, conversions, solve_sylvester as ss, utils
from hd_utils.colorize import colorize

asus_xtion_pro_f = 544.260779961

def get_phasespace_transform ():
    """
    Uses marker 0 as origin, 0->1 as x axis and 0->2 as y axis.
    Returns none if relevant markers not found.
    """
    marker_pos = ph.get_marker_positions()
    
    for i in range(3):
        if i not in marker_pos: return None
        
    # Take 0->1 as x axis, 0->2 as y axis
    x = np.asarray(marker_pos[1]) - np.asarray(marker_pos[0])
    y = np.asarray(marker_pos[2]) - np.asarray(marker_pos[0])
    x = x/nlg.norm(x)
    y = y/nlg.norm(y)
    
    z = np.cross(x,y)
    y = np.cross(z,x)
    
    tfm = np.r_[np.array([x,y,z,marker_pos[0]]).T,np.array([0,0,0,1])]
    return tfm
    
def get_ar_transform_id (depth, rgb, idm=None):    
    """
    In order to run this, ar_marker_service needs to be running.
    """
    if rospy.get_name() == '/unnamed':
        rospy.init_node('keypoints')
    
    getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)
    
    #xyz = svi.transform_pointclouds(depth, tfm)
    xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
    pc = ru.xyzrgb2pc(xyz, rgb, '/camera_link')
    
    req = MarkerPositionsRequest()
    req.pc = pc
    
    marker_tfm = {}
    res = getMarkers(req)
    for marker in res.markers.markers:
        marker_tfm[marker.id] = conversions.pose_to_hmat(marker.pose.pose)
    
    #print "Marker ids found: ", marker_tfm.keys()
    
    if not idm: return marker_tfm
    if idm not in marker_tfm: return None
    return marker_tfm[idm]


def publish_transform_markers(grabber, marker, T, from_frame, to_frame, rate=100):

    from visualization_msgs.msg import Marker
    
    print colorize("Transform : ", "yellow", True)
    print T
    
    marker_frame = "ar_frame"
    tf_pub = tf.TransformBroadcaster()
    trans = T[0:3,3] 
    rot   = tf.transformations.quaternion_from_matrix(T)
    sleeper = rospy.Rate(100)

    while True:
        try:
            r, d = grabber.getRGBD()
            ar_tfm = get_ar_transform_id (d, r, marker)
            trans_m, rot_m = conversions.hmat_to_trans_rot(ar_tfm)
            
            tf_pub.sendTransform(trans, rot, rospy.Time.now(), to_frame, from_frame)
            tf_pub.sendTransform(trans_m, rot_m, rospy.Time.now(), marker_frame, from_frame)
            
            sleeper.sleep()
        except KeyboardInterrupt:
            break    


def get_transform_kb (grabber, marker, n_tfm=3, print_shit=True):
    """
    Prompts the user to hit the keyboard whenever taking an observation.
    Switch on phasespace before this.
    """    
    tfms_ar = []
    tfms_ph = []
    
    i = 0
    while i < n_tfm:
        raw_input("Hit return when ready to take transform %i."%i)
        
        rgb, depth = grabber.getRGBD()
        ar_tfm = get_ar_transform_id(depth, rgb, marker)
        ph_tfm = get_phasespace_transform()

        if ar_tfm is None: 
            print "Could not find AR marker %i. Try again." %marker
            continue
        if ph_tfm is None:
            print "Could not correct phasespace markers. Try again."
            continue
        
        if print_shit:
            print "\n ar: "
            print ar_tfm
            print ar_tfm.dot(I_0).dot(ar_tfm.T)
            print "ph: "
            print ph_tfm
            print ph_tfm.dot(I_0).dot(ph_tfm.T)

        
        tfms_ar.append(ar_tfm)
        tfms_ph.append(ph_tfm)
        i += 1
    
    print "Found %i transforms. Calibrating..."%n_tfm
    Tas = ss.solve4(tfms_ar, tfms_ph)
    print "Done."
    
    Tcp = tfms_ar[0].dot(Tas.dot(nlg.inv(tfms_ph[0])))
    return Tcp

# TODO: Average transforms - would definitely work better
def get_transform_freq (grabber, marker, freq=0.5, n_tfm=3, print_shit=True):
    """
    Stores data at frequency provided.
    Switch on phasespace before this.
    """
    tfms_ar = []
    tfms_ph = []

    I_0 = np.eye(4)
    I_0[3,3] = 0
    
    i = 0
    
    wait_time = 5
    print "Waiting for %f seconds before collecting data."%wait_time
    time.sleep(wait_time)
    
    
    while i < n_tfm:
        print "Transform %i"%(i+1)
        rgb, depth = grabber.getRGBD()
        ar_tfm = get_ar_transform_id(depth, rgb, marker)
        ph_tfm = get_phasespace_transform()

        if ar_tfm is None: 
            print "Could not find AR marker %i." %marker
            continue
        if ph_tfm is None:
            print "Could not correct phasespace markers."
            continue
        
        if print_shit:
            print "\n ar: "
            print ar_tfm
            print ar_tfm.dot(I_0).dot(ar_tfm.T)
            print "ph: "
            print ph_tfm
            print ph_tfm.dot(I_0).dot(ph_tfm.T)
            
        
        tfms_ar.append(ar_tfm)
        tfms_ph.append(ph_tfm)
        i += 1
        time.sleep(1/freq)
    
    print "Found %i transforms. Calibrating..."%n_tfm
    Tas = ss.solve4(tfms_ar, tfms_ph)
    print "Done."
    
    T_cps = [tfms_ar[i].dot(T_ms).dot(np.linalg.inv(tfms_ph[i])) for i in xrange(len(tfms_ar))]
    trans_rots = [conversions.hmat_to_trans_rot(tfm) for tfm in T_cps]
    
    trans = np.asarray([trans for (trans, rot) in trans_rots])
    avg_trans = np.sum(trans,axis=0)/trans.shape[0]
    rots = [rot for (trans, rot) in trans_rots]
    avg_rot = avg_quaternions(np.array(rots))
    
    Tcp = conversions.trans_rot_to_hmat(avg_trans, avg_rot)
    return Tcp

from threading import Thread
from hd_logging import phasespace_logger as pl

class threadClass (Thread):    
    def run(self):
        pl.publish_phasespace_markers_ros()


def main_loop():
    rospy.init_node("phasespace_camera_calibration")
    ph.turn_phasespace_on()

    marker = 10
    n_tfm = 5
    
    grabber = cpr.CloudGrabber()
    grabber.startRGBD()
    
    Tcp = get_transform_freq(grabber, marker, n_tfm=n_tfm)

    thc = threadClass()
    thc.start()
    
    publish_transform_markers(grabber, marker, Tcp, "camera_link", "phasespace_frame")
    
    ph.turn_phasespace_off()