import numpy as np
import serial

import roslib; roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse
roslib.load_manifest('tf')
import rospy, tf
import time

from hd_utils import clouds, ros_utils as ru, conversions, utils
from cyni_cameras import cyni_cameras

import read_arduino 

np.set_printoptions(precision=5, suppress=True)

ar_lock = False
getMarkers = None
req = None
tf_l = None
arduino = None

ar_initialized = False
hydra_initialized = False
tf_initialized = False
pot_initialized = False


asus_xtion_pro_f = 544.260779961

def get_ar_marker_poses (rgb, depth, pc=None):
    """
    In order to run this, ar_marker_service needs to be running.
    """
    global getMarkers, req, ar_initialized, ar_lock
    
    while(ar_lock):
        time.sleep(0.01)
        
    ar_lock = True
    if not ar_initialized:
        #if rospy.get_name() == '/unnamed':
        #    rospy.init_node('ar_tfm')
        getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)
        req = MarkerPositionsRequest()
        ar_initialized = True
        print "AR INITIALIZED"
    
    if pc is None:
        xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
        pc = ru.xyzrgb2pc(xyz, rgb, '/camera_frame')
    
    req.pc = pc
    
    marker_tfm = {}
    res = getMarkers(req)
    for marker in res.markers.markers:
        marker_tfm[marker.id] = conversions.pose_to_hmat(marker.pose.pose)
    
    ar_lock = False
    return marker_tfm


def get_ar_markers_from_cameras (cameras, parent_frame = None, cams = None, markers=None):
    """
    The cameras here are cyni cameras.
    Returns all the ar_markers in the frame of camera 0.
    This would potentially take a while.
    Do this after all collecting data from everywhere else.
    cams -> list of cameras to use
    """
    if parent_frame is None:
        parent_frame = cameras.parent_frame

    ar_markers = {}
    
    data = cameras.get_RGBD(cams)
    for cam in data:
        tfm = cameras.get_transform_frame(cam, parent_frame)
        
        ar_cam = get_ar_marker_poses(data[cam]['rgb'], data[cam]['depth'])
        for marker in ar_cam:    
            if ar_markers.get(marker) is None:
                ar_markers[marker] = []
            ar_markers[marker].append(tfm.dot(ar_cam[marker]))
    
    for marker in ar_markers:
        ar_markers[marker] = utils.avg_transform(ar_markers[marker])
    
    if markers is None:
        return ar_markers
    else:
        return {marker:ar_markers[marker] for marker in ar_markers if marker in markers}


def get_hydra_transforms(parent_frame, hydras=None):
    """
    Returns hydra transform in hydra base frame.
    hydras is a list which contains 'left', 'right' or both 
    """
    global tf_l, hydra_initialized
    if not hydra_initialized:
        #if rospy.get_name() == '/unnamed':
        #    rospy.init_node('hydra_tfm')
        if tf_l is None:
            tf_l = tf.TransformListener()
        hydra_initialized = True
        print "HYDRA INITIALIZED"
        
    if hydras is None:
        hydras = ['left','right']
    
    hydra_transforms = {}
    for hydra in hydras:
        hydra_frame = 'hydra_%s'%hydra
        try:
            trans, rot = tf_l.lookupTransform(parent_frame, hydra_frame, rospy.Time(0))
            hydra_transforms[hydra] = conversions.trans_rot_to_hmat(trans, rot)
        except (tf.LookupException, tf.ExtrapolationException, tf.ConnectivityException):
            continue
    
    return hydra_transforms


def get_transform_frames (parent_frame, child_frame):
    """
    Gets transform between frames.
    """
    if tf_initialized is False:
        #if rospy.get_name() == '/unnamed':
        #    rospy.init_node('tf_finder')
        if tf_l is None:
            tf_l = tf.TransformListener()
        tf_initialized = True
        print "TF INITIALIZED"
        
    trans, quat = tf_l.lookupTransform(parent_frame, child_frame, rospy.Time(0))
    return conversions.trans_rot_to_hmat(trans, quat)

b = 0.0
a = (300.0-b)/30.0

def get_pot_angle ():
    """
    Get angle of gripper from potentiometer.
    """
    global pot_initialized, arduino, a, b
    if not pot_initialized:
        arduino = read_arduino.Arduino()
        pot_initialized = True
        print "POT INITIALIZED"
        
    pot_reading = arduino.get_reading()    
    return (pot_reading-b)/a
    
