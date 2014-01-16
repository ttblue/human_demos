"""
What will this script do?

simple_feedback will check in some file where extents of a box are stored in some frame.
It will consider the values in the hydra_base frame and will inform the user when he/she is out of PR2's reach.

IK_feedback will assume that the human being and the PR2 are standing at the same spot.
It will then use openrave IK based on the hydra sensor's estimate of the tool tip transform.
(Remember to convert to relevant ee-frame).

init_feedback will initialize the different kinds of feedback (maybe create the files).
simple: takes a bunch of extreme points and saves AABB.

Will only provide feedback during recording.  
"""

import cPickle, os, os.path as osp
import openravepy as opr
import subprocess
import numpy as np, numpy.linalg as nlg
import rospy, roslib

roslib.load_manifest('pocketsphinx')
from pocketsphinx.msg import Segment

roslib.load_manifest('tf')
import tf

from hd_calib.cameras import RosCameras
from hd_calib.camera_calibration import CameraCalibrator
from hd_calib.calibration_pipeline import CalibratedTransformPublisher, gripper_trans_marker_tooltip
import hd_calib.get_marker_transforms as gmt
from hd_calib import gripper_lite
from hd_utils import conversions
from hd_utils.colorize import *
from hd_utils.utils import avg_transform
from hd_utils.defaults import feedback_dir, simple_feedback_name, ik_feedback_name, \
    cam_pr2_calib_name, tfm_link_rof, tfm_gtf_ee
from hd_utils.yes_or_no import yes_or_no

s_mins = None
s_maxes = None
T_bf_hb = None
hydra_rel_tfm = {}

env = None
robot = None
manips = {}

tfl = None
tf_sleeper = None
devnull = open(os.devnull, 'wb')

class voice_alerts ():
    """
    Class to process and use voice commands.
    """
    def __init__(self):
        
        self.segment_state = None
        self.sub = rospy.Subscriber('/segment', Segment, callback=self.segment_cb)
        
    def segment_cb (self, msg):
        self.segment_state = msg.command
        blueprint(self.segment_state)
    
    def get_latest_msg (self):
        return self.segment_state

def get_avg_hydra_tfm(lr, n_avg, sleeper):
    """
    Returns averaged hydra tfms.
    """
    if not isinstance(lr, list): lr = [lr]

    avg_tfms = {h:[] for h in lr}
    
    j = 0
    while j < n_avg:
        hydra_tfms = gmt.get_hydra_transforms('hydra_base', lr)
        if not hydra_tfms:
            continue
        for h in hydra_tfms:
            avg_tfms[h].append(hydra_tfms[h])
        j += 1
        sleeper.sleep()
    for h in lr: avg_tfms[h] = avg_transform(avg_tfms[h])
 
    return avg_tfms
    

def initialize_simple (n_avg=30):
    """
    Initializes the file to load from for simple feedback.
    
    Creates AABB out of specified points.
    """
    
    if rospy.get_name() == '/unnamed':
        rospy.init_node('simple_feedback_init', anonymous=True)
    sleeper = rospy.Rate(30)
    
    yellowprint("You will be asked to move gripper around the feasibility zone.")
    
    lr = raw_input(colorize("Please enter l/r based on which gripper you are using for initialization: ",'yellow',True))
    lr_long = {'l':'left', 'r':'right'}[lr]


    aabb_points = []

    raw_input(colorize("Place the gripper at bottom-left of reachable region closer to you.",'blue',True))
    attempts = 10
    while attempts > 0:
        try:
            blc_tfm = get_avg_hydra_tfm(lr_long, n_avg, sleeper)[lr_long]
            print blc_tfm
            aabb_points.append(blc_tfm[0:3,3])
            break
        except Exception as e:
            print e
            attempts -= 1
            yellowprint("Failed attempt. Trying %i more times."%attempts)
            sleeper.sleep()
    if attempts == 0:
        redprint("Could not find hydra.")
    
    raw_input(colorize("Place the gripper at bottom-right of reachable region closer to you.",'blue',True))
    attempts = 10
    while attempts > 0:
        try:
            brc_tfm = get_avg_hydra_tfm(lr_long, n_avg, sleeper)[lr_long]
            print brc_tfm[0:3,3]
            aabb_points.append(brc_tfm[0:3,3])
            break
        except:
            attempts -= 1
            yellowprint("Failed attempt. Trying %i more times."%attempts)
            sleeper.sleep()
    if attempts == 0:
        redprint("Could not find hydra.")
        
    raw_input(colorize("Place the gripper at top-left of reachable region closer to you.",'blue',True))
    attempts = 10
    while attempts > 0:
        try:
            tlc_tfm = get_avg_hydra_tfm(lr_long, n_avg, sleeper)[lr_long]
            print tlc_tfm[0:3,3]
            aabb_points.append(tlc_tfm[0:3,3])
            break
        except:
            attempts -= 1
            yellowprint("Failed attempt. Trying %i more times."%attempts)
            sleeper.sleep()
    if attempts == 0:
        redprint("Could not find hydra.")
    
    raw_input(colorize("Place the gripper at top-right of reachable region closer to you.",'blue',True))
    attempts = 10
    while attempts > 0:
        try:
            trc_tfm = get_avg_hydra_tfm(lr_long, n_avg, sleeper)[lr_long]
            print trc_tfm[0:3,3]
            aabb_points.append(trc_tfm[0:3,3])
            break
        except:
            attempts -= 1
            yellowprint("Failed attempt. Trying %i more times."%attempts)
            sleeper.sleep()
    if attempts == 0:
        redprint("Could not find hydra.")
         
    raw_input(colorize("Place the gripper as far from you as possible in the reachable region.",'blue',True))
    attempts = 10
    while attempts > 0:
        try:
            far_tfm = get_avg_hydra_tfm(lr_long, n_avg, sleeper)[lr_long]
            print far_tfm[0:3,3]
            aabb_points.append(far_tfm[0:3,3])
            break
        except:
            attempts -= 1
            yellowprint("Failed attempt. Trying %i more times."%attempts)
            sleeper.sleep()
    if attempts == 0:
        redprint("Could not find hydra.")
    
    buffer = 0.05 # a little buffer for reachability
    aabb_points = np.asarray(aabb_points)
    mins = aabb_points.min(axis=0) - buffer
    maxes = aabb_points.max(axis=0) + buffer
    aabb_data = {'mins':mins, 'maxes':maxes}

    yellowprint("Saving AABB in file...")
    aabb_file = osp.join(feedback_dir, simple_feedback_name)
    with open(aabb_file,'w') as fh: cPickle.dump(aabb_data, fh)
    yellowprint("Saved!")


def check_simple (lr, tfm):
    """
    Checks if tfm is valid.
    lr is not needed but is helpful for function generalization.
    """
    global s_mins, s_maxes
    
    if s_mins == None or s_maxes == None:
        aabb_file = osp.join(feedback_dir, simple_feedback_name)
        with open(aabb_file,'r') as fh: data = cPickle.load(fh)
        s_mins, s_maxes = data['mins'], data['maxes']
    
    pos = tfm[0:3,3]
    return all(pos >= s_mins) and all(pos <= s_maxes)


def get_transform (parent_frame, child_frame, n_attempts = None):
    """
    Wrapper for getting tf transform.
    """
    global tfl, tf_sleeper
    
    if tfl is None:
        tfl = tf.TransformListener()
        tf_sleeper = rospy.Rate(30)

    while n_attempts is None or n_attempts > 0:
        try:
            trans, rot = tfl.lookupTransform(parent_frame, child_frame, rospy.Time(0))
            tfm = conversions.trans_rot_to_hmat(trans, rot)
            return tfm
        except (tf.LookupException, tf.ExtrapolationException, tf.ConnectivityException):
            if n_attempts is not None:
                n_attempts -= 1
            tf_sleeper.sleep()

    redprint("Unable to find transform from %s to %s."%(parent_frame, child_frame))
    return None


def initialize_ik (calib_file, n_tfm=5, n_avg=30):
    """
    Initializes the file to load from for ik feedback.
    
    First calibrates between PR2 and workspace camera.
    Finally ends up with a calibration from the PR2 base with the hydra_base.
    Can use hydra_info to reach tool tip from movement data. 
    """
    
    if rospy.get_name() == '/unnamed':
        rospy.init_node('ik_feedback_init')
    
    cameras = RosCameras()
    tfm_pub = CalibratedTransformPublisher(cameras)

    yellowprint("Start up the overhead camera as camera1 and PR2's camera as camera2.")
    raw_input(colorize("Hit enter when the PR2 is ready at the workstation.",'yellow',True))

    c1_frame = 'camera1_link'
    h_frame = 'hydra_base'

    # Find tfm from bf to camera1_link    
    greenprint("Calibrating cameras.")
    cam_calib = CameraCalibrator(cameras)
    done = False
    while not done:
        cam_calib.calibrate(n_obs=n_tfm, n_avg=n_avg)
        if not cam_calib.calibrated:
            redprint("Camera calibration failed.")
            cam_calib.reset_calibration()
        else:
            # Add camera to camera tfm and camera to PR2 tfm
            tfm_cam = cam_calib.get_transforms()[0]
            tfm_pr2 = {'parent':'base_footprint', 'child':c1_frame}
            tfm_bf_c2l = get_transform('base_footprint','camera_link')
            tfm_pr2['tfm'] = tfm_bf_c2l.dot(np.linalg.inv(tfm_cam['tfm']))

#             tfm_pub.add_transforms(cam_calib.get_transforms())
            tfm_pub.add_transforms([tfm_cam, tfm_pr2])
            print tfm_pub.transforms
            
            if yes_or_no("Are you happy with the calibration? Check RVIZ."):
                done = True
                tfm_pub.save_calibration(cam_pr2_calib_name)
                greenprint("Cameras calibrated.")
            else:
                yellowprint("Calibrating cameras again.")
                cam_calib.reset_calibration()        

    tfm_c1_h = None
    # Now, find transform between base_footprint and hydra base
    with open(calib_file,'r') as fh: calib_data = cPickle.load(fh)
    for tfm in calib_data['transforms']:
        if tfm['parent'] == c1_frame or tfm['parent'] == '/' + c1_frame:
            if tfm['child'] == h_frame or tfm['child'] == '/' + h_frame:
                tfm_c1_h = tfm['tfm']

    if tfm_c1_h is None:
        redprint("Hydra calib info not found in %s."%calib_file)
    
    grippers = calib_data['grippers']
    assert 'l' in grippers.keys() and 'r' in grippers.keys(), 'Calibration does not have both grippers.'
    
    gprs = {}
    for lr,gdata in grippers.items():
        gr = gripper_lite.GripperLite(lr, gdata['ar'], trans_marker_tooltip=gripper_trans_marker_tooltip[lr])
        gr.reset_gripper(lr, gdata['tfms'], gdata['ar'], gdata['hydra'])
        gprs[lr] = gr
    
    ik_file_data = {}
    ik_file_data['pr2'] = tfm_pr2['tfm'].dot(tfm_c1_h)
    lr_long = {'l':'left','r':'right'}
    for lr in gprs:
        ik_file_data[lr_long[lr]] = gprs[lr].get_rel_transform(lr_long[lr], 'tool_tip')    

    
    yellowprint("Saving IK feedback data in file...")
    ik_file = osp.join(feedback_dir, ik_feedback_name)
    with open(ik_file,'w') as fh: cPickle.dump(ik_file_data, fh)
    yellowprint("Saved!")


def check_ik (lr, tfm):
    """
    Checks if tfm is valid.
    Uses openrave IK.
    """
    global T_bf_hb, hydra_rel_tfm, env, robot, manips
    
    # Initialize a bunch of parameters
    if T_bf_hb is None or not hydra_rel_tfm:
        ik_file = osp.join(feedback_dir, ik_feedback_name)
        with open(ik_file,'r') as fh: data = cPickle.load(fh)
        T_bf_hb = data['pr2']
        hydra_rel_tfm['right'] = data['right']
        hydra_rel_tfm['left'] = data['left']
    
    if env is None:
        env = opr.Environment()
        env.Load('robots/pr2-beta-static.zae')
        robot = env.GetRobots()[0]
        manips = {'left':robot.GetManipulator('leftarm'),
                  'right':robot.GetManipulator('rightarm')}
        # Maybe add table
    ee_tfm = T_bf_hb.dot(tfm).dot(hydra_rel_tfm[lr]).dot(tfm_gtf_ee)
    
    iksol = manips[lr].FindIKSolution(opr.IkParameterization(ee_tfm, opr.IkParameterizationType.Transform6D), 
                                      opr.IkFilterOptions.IgnoreEndEffectorEnvCollisions)
    
    return iksol != None


def feedback_loop (method='simple'):
    """
    Gives feedback based on method and hydra data.
    """

    assert method in ['simple','ik'], "Feedback method %s not implemented."%method

    rospy.init_node("%s_feedback_node"%method)
    
    cmd_checker = voice_alerts()
    check_func = {'simple':check_simple, 'ik':check_ik}[method]    
    hydras = ['left','right']
    
    warn_rate = rospy.Rate(1)
    cmd_sleeper = rospy.Rate(10)
    check_rate = rospy.Rate(30)
    sleeper = rospy.Rate(30)
    n_avg = 5 
    
    # Hydra seems to have some issues to begin with.
    # So let's try that first.
    while not rospy.is_shutdown():
        try:
            get_avg_hydra_tfm(hydras, 1, check_rate)
            break
        except:
            continue
    
    yellowprint("Got first hydra message.")
    
    while not rospy.is_shutdown():
        # Wait for begin recording        
        while not rospy.is_shutdown():
            status = cmd_checker.get_latest_msg()
            if  status in ["begin recording"]:
                break
            cmd_sleeper.sleep()
            
        # Check feasibility while waiting for cancel recording
        time_thresh = 2.0
        tstart = rospy.Time.now().to_sec()
        last_time = {h:tstart for h in hydras}
        while not rospy.is_shutdown():
            status = cmd_checker.get_latest_msg()
            # Done with session
            if  status in ["cancel recording","finish recording"]:
                break
            hydra_tfms = get_avg_hydra_tfm(hydras, n_avg, check_rate)
            time_now = rospy.Time.now().to_sec()
  
            for lr in hydras:
                if check_func(lr, hydra_tfms[lr]):
                    last_time[lr] = time_now
    
            hydra_viol = ''
            for lr in hydras:
                if time_now - last_time[lr] > time_thresh:
                    if hydra_viol:
                        hydra_viol += ' and '
                    hydra_viol += lr
            
            if hydra_viol:
                subprocess.call("espeak -v en 'Hydra %s invalid.'"%hydra_viol, stdout=devnull, stderr=devnull, shell=True)
                warn_rate.sleep()

        sleeper.sleep()
    
    
if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", help="Choose from simple or ik.", default="simple", type=str)
    parser.add_argument("--initialize", help="Initialize the feedback files.", action="store_true",default=False)
    parser.add_argument("--calib_file", help="Path to calibration file for IK init.", type=str)
    parser.add_argument("--ntfm", help="Number of transforms for initializing IK.", type=int, default=5)
    parser.add_argument("--navg", help="Number of transforms to average for initializing simple/IK.", type=int, default=30)
        
    args = parser.parse_args()

    if args.initialize:
        if args.method == 'ik':
            initialize_ik(args.calib_file, args.ntfm, args.navg)
        elif args.method == 'simple':
            initialize_simple(args.navg)
    else:
        feedback_loop(args.method)
            