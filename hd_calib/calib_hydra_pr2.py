"""
Calibrates the pr2's hand with the hydra : similar to hydra--ar marker calibration.
"""

from __future__ import division

import roslib
# roslib.load_manifest('calib_hydra_pr2')
import rospy
roslib.load_manifest('tf')
import tf; 

import argparse
import numpy as np, numpy.linalg as nlg
import scipy as scp, scipy.optimize as sco
import cPickle
from hd_utils.colorize import colorize
from hd_utils import conversions, solve_sylvester as ss
from hd_utils.utils import avg_transform

np.set_printoptions(precision=5, suppress=True)

def get_transforms(arm, hydra, n_tfm , n_avg):
    """
    Returns a tuple of two list of N_TFM transforms, each
    the average of N_AVG transforms
    corresponding to the left/ right arm of the pr2 and 
    the hydra paddle of the HYDRA ('right' or 'left') side.
    
    return arm_tfms, hydra_tfms
    """

    pr2_frame = 'base_footprint'
    assert arm == 'right' or 'left'
    arm_frame = '%s_gripper_tool_frame' % {'right':'r', 'left':'l'}[arm]

    hydra_frame = 'hydra_base'
    assert hydra == 'right' or hydra == 'left'
    paddle_frame = 'hydra_%s' % hydra

    tf_sub = tf.TransformListener()

    I_0 = np.eye(4)
    I_0[3, 3] = 0
    
    gripper_tfms = []
    hydra_tfms = []

    for i in xrange(n_tfm):
        raw_input(colorize("Transform %d of %d : Press return when ready to capture transforms" % (i, n_tfm - 1), "red", True))
        
        # # transforms which need to be averaged.
        gtfms = []
        htfms = []
        
        sleeper = rospy.Rate(30)        
        for j in xrange(n_avg):            
            print colorize('\tGetting averaging transform : %d of %d ...' % (j, n_avg - 1), "blue", True)

            gtrans, grot = tf_sub.lookupTransform(pr2_frame, arm_frame, rospy.Time(0))
            gtfms.append(conversions.trans_rot_to_hmat(gtrans, grot))
            htrans, hrot = tf_sub.lookupTransform(hydra_frame, paddle_frame, rospy.Time(0))
            htfms.append(conversions.trans_rot_to_hmat(htrans, hrot))
            sleeper.sleep()
            
        gripper_tfms.append(avg_transform(gtfms))
        hydra_tfms.append(avg_transform(htfms))

    return (gripper_tfms, hydra_tfms)
    

def publish_tf(T, from_frame, to_frame):
    print colorize("Transform : ", "yellow", True)
    print T
    tf_pub = tf.TransformBroadcaster()
    trans = T[0:3, 3] 
    rot = tf.transformations.quaternion_from_matrix(T)
    sleeper = rospy.Rate(100)

    while True:
        tf_pub.sendTransform(trans, rot, rospy.Time.now(), to_frame, from_frame)
        sleeper.sleep()


if __name__ == '__main__':
    rospy.init_node('calib_hydra_pr2')
    
    parser = argparse.ArgumentParser(description="Hydra Kinect Calibration")
    parser.add_argument('--arm', help="in {'right', 'left'} : the pr2 arm to track.", required=True)  
    parser.add_argument('--hydra', help="in {'right', 'left'} : the hydra handle to track.", required=True)
    parser.add_argument('--n_tfm', help="number of transforms to use for calibration.", type=int, default=5)
    parser.add_argument('--n_avg', help="number of estimates of  transform to use for averaging.", type=int, default=5)
    parser.add_argument('--publish_tf', help="whether to publish the transform between hydra_base and camera_link", default=True)
    vals = parser.parse_args()

    arm_tfms, hydra_tfms = get_transforms(vals.arm, vals.hydra, vals.n_tfm, vals.n_avg)
    
    if vals.publish_tf:
        T_ms = ss.solve4(arm_tfms, hydra_tfms)
        T_chs = [arm_tfms[i].dot(T_ms).dot(np.linalg.inv(hydra_tfms[i])) for i in xrange(len(arm_tfms))]
        trans_rots = [conversions.hmat_to_trans_rot(tfm) for tfm in T_chs]
        
        trans = np.asarray([trans for (trans, rot) in trans_rots])
        avg_trans = np.sum(trans, axis=0) / trans.shape[0]
        rots = [rot for (trans, rot) in trans_rots]
        avg_rot = avg_quaternions(np.array(rots))
        
        T_ch = conversions.trans_rot_to_hmat(avg_trans, avg_rot)
        T_arm_sensor = [np.linalg.inv(arm_tfms[i]).dot(T_ch).dot(hydra_tfms[i]) for i in xrange(len(arm_tfms))]
        trans_rots = [conversions.hmat_to_trans_rot(tfm) for tfm in T_arm_sensor]
        trans = np.asarray([trans for (trans, rot) in trans_rots])
        avg_trans = np.sum(trans, axis=0) / trans.shape[0]
        rots = [rot for (trans, rot) in trans_rots]
        avg_rot = avg_quaternions(np.array(rots))
        T_as = conversions.trans_rot_to_hmat(avg_trans, avg_rot)

        cPickle.dump(T_as, open('T_as', 'wb'))
        publish_tf(T_ch, 'base_footprint', 'hydra_base')
        # arm_frame  = '%s_gripper_tool_frame' % {'right':'r', 'left':'l'}[vals.arm]
        # sensor_frame = 'hydra_calib'
        # publish_tf(T_as, arm_frame, sensor_frame)
