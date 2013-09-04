#!/usr/bin/python
import argparse
import time
import numpy as np, numpy.linalg as nlg
import scipy as scp, scipy.linalg as sclg, scipy.optimize as sco
import roslib; roslib.load_manifest("tf")
import rospy, tf
import roslib; roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse
import subprocess

import cloudprocpy as cpr
import phasespace as ph

from hd_utils import clouds, ros_utils as ru, conversions, solve_sylvester as ss, utils
from hd_utils.colorize import colorize

asus_xtion_pro_f = 544.260779961

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


def publish_transform_markers(grabber, marker, T, from_frame, to_frame, rate=100):

    from visualization_msgs.msg import Marker
    from sensor_msgs.msg import PointCloud2
    
    print colorize("Transform : ", "yellow", True)
    print T
    
    marker_frame = "ar_frame"
    tf_pub = tf.TransformBroadcaster()
    pc_pub = rospy.Publisher("camera_points", PointCloud2)
    
    trans = T[0:3,3] 
    rot   = tf.transformations.quaternion_from_matrix(T)
    sleeper = rospy.Rate(10)

    while True:
        try:
            r, d = grabber.getRGBD()
            ar_tfm = get_ar_transform_id (d, r, marker)
            
            pc = ru.xyzrgb2pc(clouds.depth_to_xyz(d, asus_xtion_pro_f), r, from_frame)
            pc_pub.publish(pc)
            
            tf_pub.sendTransform(trans, rot, rospy.Time.now(), to_frame, from_frame)
            
            try:
                trans_m, rot_m = conversions.hmat_to_trans_rot(ar_tfm)
                tf_pub.sendTransform(trans_m, rot_m, rospy.Time.now(), marker_frame, from_frame)
            except:
                pass
            
            sleeper.sleep()
        except KeyboardInterrupt:
            break    

def publish_transform(T, from_frame, to_frame, rate=100):
    
    print colorize("Transform : ", "yellow", True)
    print T
    
    tf_pub = tf.TransformBroadcaster()
    
    trans, rot = conversions.hmat_to_trans_rot(T)
    sleeper = rospy.Rate(10)

    while True:
        try:            
            tf_pub.sendTransform(trans, rot, rospy.Time.now(), to_frame, from_frame)
            sleeper.sleep()
        except KeyboardInterrupt:
            break    


# Incorporate averaging.
def get_transform_kb (grabber, marker, n_avg=10, n_tfm=3, print_shit=True):
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

def get_transform_ros(marker, n_tfm, n_avg, freq=None):
    """
    Returns a tuple of two list of N_TFM transforms, each
    the average of N_AVG transforms
    corresponding to the AR marker with ID = MARKER and 
    the phasespace markers.
    """

    camera_frame = 'camera_depth_optical_frame'
    marker_frame = 'ar_marker_%d'%marker
    ps_frame = 'ps_marker_transform'

    tf_sub = tf.TransformListener()

    I_0 = np.eye(4)
    I_0[3,3] = 0
    
    ar_tfms    = []
    ph_tfms = []

    wait_time = 5
    print "Waiting for %f seconds before collecting data."%wait_time
    time.sleep(wait_time)

    sleeper = rospy.Rate(30)
    for i in xrange(n_tfm+1):
        if freq is None:
            raw_input(colorize("Transform %d of %d : Press return when ready to capture transforms"%(i, n_tfm), "red", True))
        else: 
            print colorize("Transform %d of %d."%(i, n_tfm), "red", True)
        ## transforms which need to be averaged.
        ar_tfm_avgs = []
        ph_tfm_avgs = []
        
        for j in xrange(n_avg):            
            print colorize('\tGetting averaging transform : %d of %d ...'%(j,n_avg-1), "blue", True)
            
            mtrans, mrot, ptrans, prot = None, None, None, None
            while ptrans == None or mtrans == None:
                mtrans, mrot = tf_sub.lookupTransform(camera_frame, marker_frame, rospy.Time(0))
                ptrans, prot = tf_sub.lookupTransform(ph.PHASESPACE_FRAME, ps_frame, rospy.Time(0))
                sleeper.sleep()

            ar_tfm_avgs.append(conversions.trans_rot_to_hmat(mtrans,mrot))
            ph_tfm_avgs.append(conversions.trans_rot_to_hmat(ptrans,prot))
            
        ar_tfm = utils.avg_transform(ar_tfm_avgs)
        ph_tfm = utils.avg_transform(ph_tfm_avgs)

#         print "\nar:"
#         print ar_tfm
#         print ar_tfm.dot(I_0).dot(ar_tfm.T)
#         print "h:"
#         print ph_tfm
#         print ph_tfm.dot(I_0).dot(ph_tfm.T), "\n"
                
        ar_tfms.append(ar_tfm)
        ph_tfms.append(ph_tfm)
        if freq is not None:
            time.sleep(1/freq)

        
    print "Found %i transforms. Calibrating..."%n_tfm
    Tas = ss.solve4(ar_tfms, ph_tfms)
    print "Done."
    
    T_cps = [ar_tfms[i].dot(Tas).dot(np.linalg.inv(ph_tfms[i])) for i in xrange(len(ar_tfms))]
    return utils.avg_transform(T_cps)


def get_transform_freq (grabber, marker, freq=0.5, n_avg=10, n_tfm=3, print_shit=False):
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
    
    
    avg_freq = 30
    while i < n_tfm:
        print "Transform %i"%(i+1)
        
        j = 0
        ar_tfms = []
        ph_tfms = []
        while j < n_avg:
            rgb, depth = grabber.getRGBD()
            ar_tfm = get_ar_transform_id(depth, rgb, marker)
            ph_tfm = ph.marker_transform(0,1,2, ph.get_marker_positions())

            if ar_tfm is None: 
                print colorize("Could not find AR marker %i."%marker,"red",True)
                continue
            if ph_tfm is None:
                print colorize("Could not correct phasespace markers.", "red", True)
                continue

            print colorize("Got transform %i for averaging."%(j+1), "blue", True)
            ar_tfms.append(ar_tfm)
            ph_tfms.append(ph_tfm)
            j += 1
            time.sleep(1/avg_freq)
        
        ar_avg_tfm = utils.avg_transform(ar_tfms)
        ph_avg_tfm = utils.avg_transform(ph_tfms)
        
        if print_shit:
            print "\n ar: "
            print ar_avg_tfm
            print ar_avg_tfm.dot(I_0).dot(ar_avg_tfm.T)
            print "ph: "
            print ph_avg_tfm
            print ph_avg_tfm.dot(I_0).dot(ph_avg_tfm.T)
            
        
        tfms_ar.append(ar_avg_tfm)
        tfms_ph.append(ph_avg_tfm)
        i += 1
        time.sleep(1/freq)
    
    print "Found %i transforms. Calibrating..."%n_tfm
    Tas = ss.solve4(tfms_ar, tfms_ph)
    print "Done."
    
    T_cps = [tfms_ar[i].dot(Tas).dot(np.linalg.inv(tfms_ph[i])) for i in xrange(len(tfms_ar))]
    return utils.avg_transform(T_cps)

from threading import Thread
from hd_visualization import visualize_phasespace as vp

class threadClass (Thread):    
    def run(self):
        vp.publish_phasespace_markers_ros()

# Use command line arguments
# Before staring this, start up hd_visualization.visualize_ar_ps
# OR, if you have roslaunch, then run the ar marker and phasespace stuff separately
if __name__=='__main__':
#     global getMarkers

    parser = argparse.ArgumentParser(description="Phasespace Kinect Calibration")
    parser.add_argument('--marker', help="AR marker id to track.", required=True, type=int)
    parser.add_argument('--n_tfm', help="number of transforms to use for calibration.", type=int, default=5)
    parser.add_argument('--n_avg', help="number of estimates of transform to use for averaging.", type=int, default=10)
    args = parser.parse_args()
    
    #subprocess.call("killall XnSensorServer", shell=True)
    rospy.init_node("phasespace_camera_calibration")    
    
#     ph.turn_phasespace_off()
#     thc = threadClass()
#     thc.start()

    #ph.turn_phasespace_on()

#     marker = 10
#     n_tfm = 5
#     n_avg = 10
#     
#     getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)
#     
#     grabber = cpr.CloudGrabber()
#     grabber.startRGBD()
    
    Tcp = get_transform_ros(args.marker, args.n_tfm, args.n_avg)
    
    raw_input("Kill static transform and press return.")
    publish_transform(Tcp, "camera_depth_optical_frame", ph.PHASESPACE_FRAME)
    