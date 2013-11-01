# script to load and run a kalman filter on recorded demos.
from __future__ import division

import roslib
import rospy, rosbag
roslib.load_manifest("tf")
import tf
from   sensor_msgs.msg import PointCloud2
from   geometry_msgs.msg import PoseStamped

import numpy as np
import os, os.path as osp
import cPickle as cp
import scipy.linalg as scl
import math
import matplotlib.pylab as plt

from hd_utils.colorize import colorize
from hd_utils.conversions import *
from hd_utils.utils import *
from hd_utils.defaults import tfm_link_rof
from hd_track.kalman import kalman
from hd_track.kalman import smoother
from hd_track.kalman_tuning import state_from_tfms_no_velocity
from hd_track.streamer import streamize
from hd_track.stream_pc import streamize_pc
from hd_visualization.ros_vis import draw_trajectory 

hd_path = os.getenv('HD_DIR')
if hd_path is None:
    hd_path = '/home/henrylu/henry_sandbox/human_demos'


def load_covariances():
    """
    Load the noise covariance matrices:
    """
    covar_mats   =  cp.load(open(hd_path + '/hd_track/data/nodup-covars-1.cpickle'))
    ar_covar     =  1e2*covar_mats['kinect']
    motion_covar =  1e-3*covar_mats['process']
    hydra_covar  =  1e-3*covar_mats['hydra']

    return (motion_covar, ar_covar, hydra_covar)


def load_data(fname = 'demo1.data'):
    """
    return cam1, cam2, hydra transform data.
    """
    fname = osp.join(hd_path + '/hd_data/demos/obs_data', fname)
    dat   = cp.load(open(fname))
    
    cam1_ts  = np.array([tt[1] for tt in dat['camera1']])  ## time-stamps
    cam1_tfs = [tt[0] for tt in dat['camera1']]  ## transforms
    
    cam2_ts  = np.array([tt[1] for tt in dat['camera2']])  ## time-stamps
    cam2_tfs = [tt[0] for tt in dat['camera2']]  ## transforms

    hydra_ts  = np.array([tt[1] for tt in dat['hydra']])  ## time-stamps
    hydra_tfs = [tt[0] for tt in dat['hydra']]  ## transforms
    
    return (cam1_ts, cam1_tfs, cam2_ts, cam2_tfs, hydra_ts, hydra_tfs)


def relative_time_streams(fname, freq):
    c1_ts, c1_tfs, c2_ts, c2_tfs, hy_ts, hy_tfs = load_data(fname)
    dt =1/freq
    
    tmin = min(np.min(c1_ts), np.min(c2_ts), np.min(hy_ts))
    tmax = max(np.max(c1_ts), np.max(c2_ts), np.max(hy_ts))

    ## calculate the number of time-steps for the kalman filter.    
    nsteps = int(math.ceil((tmax-tmin)/dt))
    
    ## get rid of absolute time, put the three data-streams on the same time-scale
    c1_ts -= tmin
    c2_ts -= tmin
    hy_ts -= tmin
    
    ar1_strm = streamize(c1_tfs, c1_ts, freq, avg_transform)
    ar2_strm = streamize(c2_tfs, c2_ts, freq, avg_transform)
    hy_strm  = streamize(hy_tfs, hy_ts, freq, avg_transform)
    
    return tmin, tmax, nsteps, ar1_strm, ar2_strm, hy_strm


def get_first_state(dt, c1_ts, c1_tfs, c2_ts, c2_tfs, hy_ts, hy_tfs):
    """
    Return the first state (mean, covar) to initialize the kalman filter with.
    Assumes that the time-stamps start at a common zero (are on the same time-scale).
    
    Returns a state b/w t=[0, dt]
    
    Gives priority to AR-markers. If no ar-markers are found in [0,dt], it returns
    hydra's estimate but with larger covariance.
    """
    
    ar1 = [c1_tfs[i] for i in xrange(len(c1_ts)) if c1_ts[i] <= dt]
    ar2 = [c2_tfs[i] for i in xrange(len(c2_ts)) if c2_ts[i] <= dt]
    hy =  [hy_tfs[i] for i in xrange(len(hy_ts)) if hy_ts[i] <= dt] 
    
    if ar1 != [] or ar2 != []:
        ar1.extend(ar2)
        x0 =  state_from_tfms_no_velocity([avg_transform(ar1)])
        I3 = np.eye(3)
        S0 = scl.block_diag(1e-3*I3, 1e-2*I3, 1e-3*I3, 1e-3*I3)
    else:
        assert len(hy)!=0, colorize("No transforms found for KF initialization. Aborting.", "red", True)
        x0 = state_from_tfms_no_velocity([avg_transform(hy)])
        I3 = np.eye(3)
        S0 = scl.block_diag(1e-1*I3, 1e-1*I3, 1e-2*I3, 1e-2*I3)
    return (x0, S0)
        


def setup_kalman(fname, freq=30.):
    c1_ts, c1_tfs, c2_ts, c2_tfs, hy_ts, hy_tfs = load_data(fname)
    motion_var, ar_var, hydra_var = load_covariances()
    dt = 1/freq
    
    tmin = min(np.min(c1_ts), np.min(c2_ts), np.min(hy_ts))
    tmax = max(np.max(c1_ts), np.max(c2_ts), np.max(hy_ts))

    ## calculate the number of time-steps for the kalman filter.    
    nsteps = int(math.ceil((tmax-tmin)/dt))
    
    ## get rid of absolute time, put the three data-streams on the same time-scale
    c1_ts -= tmin
    c2_ts -= tmin
    hy_ts -= tmin
    
    # initialize KF:
    x0, S0 = get_first_state(dt, c1_ts, c1_tfs, c2_ts, c2_tfs, hy_ts, hy_tfs)
    KF     = kalman()
    
    ## ===> assumes that the variance for ar1 and ar2 are the same!!!
    hydra_var_vel = np.zeros((6, 6))
    hydra_var_vel[3:6, 3:6] = hydra_var[3:6, 3:6]
    hydra_var_vel[0:3, 0:3] = 25e-8 * np.eye(3)
    KF.init_filter(0,x0, S0, motion_var, hydra_var_vel, ar_var, ar_var)
    
    ar1_strm = streamize(c1_tfs, c1_ts, freq, avg_transform)
    ar2_strm = streamize(c2_tfs, c2_ts, freq, avg_transform)
    hy_strm  = streamize(hy_tfs, hy_ts, freq, avg_transform)
    
    return (KF, nsteps, tmin, ar1_strm, ar2_strm, hy_strm)


def soft_next(stream):
    """
    Does not throw a stop-exception if a stream ends. Instead returns none.
    """
    ret = None
    try:
        ret = stream.next()
    except :
        pass
    return ret
        


def run_kalman_filter(fname, freq=30.):
    """
    Runs the kalman filter
    """
    dt = 1/freq
    KF, nsteps, tmin, ar1_strm, ar2_strm, hy_strm = setup_kalman(fname, freq)
    
    ## run the filter:
    mu,S = [], []
    for i in xrange(nsteps):
        KF.register_observation(dt*(i+1), soft_next(ar1_strm), soft_next(ar2_strm), soft_next(hy_strm))
        mu.append(KF.x_filt)
        S.append(KF.S_filt)

    A, R = KF.get_motion_mats(dt)
    return nsteps, tmin, mu, S, A, R

def get_cam_transform():
    calib_file = osp.join(hd_path + '/hd_data/calib/calib_cam2')
    dat = cp.load(open(calib_file))
    T_l1l2 = dat['transforms'][0]['tfm']

    return np.linalg.inv(tfm_link_rof).dot(T_l1l2.dot(tfm_link_rof)) 

def publish_static_tfm(parent_frame, child_frame, tfm):
    import thread
    tf_broadcaster = tf.TransformBroadcaster()
    trans, rot = hmat_to_trans_rot(tfm)
    def spin_pub():
        sleeper = rospy.Rate(100)
        while not rospy.is_shutdown():
            tf_broadcaster.sendTransform(trans, rot,
                                     rospy.Time.now(),
                                     child_frame, parent_frame)
            sleeper.sleep()
    thread.start_new_thread(spin_pub, ())


def open_frac(th):
    thmax = 33
    return th/thmax


def plot_kalman(X_kf, X_ks, X_ar1, vs_ar1, X_ar2, vs_ar2, X_hy, vs_hy):
    """
    
    """

    to_plot=[0,1,2,6,7,8]
    axlabels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw', 'v_roll', 'v_pitch', 'v_yaw']
    for i in to_plot:
        plt.subplot(4,3,i+1)
        plt.plot(X_kf[i,:], label='filter')
        plt.plot(X_ks[i,:], label='smoother')
        plt.plot(vs_ar1, X_ar1[i,:], '.', label='camera1')
        plt.plot(vs_ar2, X_ar2[i,:], '.', label='camera2')
        plt.plot(vs_hy, X_hy[i,:], '.', label='hydra')
        plt.ylabel(axlabels[i])
        plt.legend()


            
if __name__ == '__main__':
    demo_num = 1
    freq     = 30.

    data_dir = os.getenv('HD_DATA_DIR') 
    #bag = rosbag.Bag(osp.join(data_dir,'demos/recorded/demo'+str(demo_num)+'.bag'))
    rospy.init_node('viz_demos')
    pub = rospy.Publisher('/point_cloud1', PointCloud2)
    pub2= rospy.Publisher('/point_cloud2', PointCloud2)
    
    ## publishers for unfiltered-data:
    c1_tfm_pub    = rospy.Publisher('/ar1_estimate', PoseStamped)
    c2_tfm_pub    = rospy.Publisher('/ar2_estimate', PoseStamped)
    hydra_tfm_pub = rospy.Publisher('/hydra_estimate', PoseStamped)
    _, _, _, ar1_strm, ar2_strm, hy_strm = relative_time_streams('demo'+str(demo_num)+'.data', freq)

    ## run the kalman filter:
    nsteps, tmin, F_means,S,A,R = run_kalman_filter('demo'+str(demo_num)+'.data', freq)
    S_means = smoother(A, R, F_means, S)
    #print S_means[0]
    X_kf = np.array(F_means)
    X_kf = np.reshape(X_kf, (X_kf.shape[0], X_kf.shape[1])).T

    X_ks = np.array(S_means)
    X_ks = np.reshape(X_ks, (X_ks.shape[0], X_ks.shape[1])).T

    T_filt = state_to_hmat(S_means)
    
    ## load the potentiometer-angle stream:
    pot_data = cp.load(open(osp.join(data_dir, 'demos/obs_data/demo' +str(demo_num)+'.data')))['pot_angles']
    ang_ts   = np.array([tt[1] for tt in pot_data])  ## time-stamps
    ang_vals = [open_frac(tt[0]) for tt in pot_data]  ## angles
    ang_strm = streamize(ang_vals, ang_ts, freq, lambda x : x[-1], tmin)

    cam1_frame_id = '/camera1_rgb_optical_frame'
    cam2_frame_id = '/camera2_rgb_optical_frame'

    ## get the relative-transforms between the cameras:
    cam_tfm  = get_cam_transform()
    publish_static_tfm(cam1_frame_id, cam2_frame_id, cam_tfm)

    handles = []
    
    ## frame of the filter estimate:
    sleeper = rospy.Rate(freq)
    
    vs_ar1 = []
    vs_ar2 = []
    vs_hy = []

    Ts_ar1 = []
    Ts_ar2 = []
    Ts_hy = []

    for i in xrange(nsteps):
        ar1_est = soft_next(ar1_strm)
        ar2_est = soft_next(ar2_strm)
        hy_est = soft_next(hy_strm)

        if ar1_est != None:
            Ts_ar1.append(ar1_est)
            vs_ar1.append(i)
        if ar2_est != None:
            Ts_ar2.append(ar2_est)
            vs_ar2.append(i)
        if hy_est != None:
            Ts_hy.append(hy_est)
            vs_hy.append(i)

    X_ar1 = state_from_tfms_no_velocity(Ts_ar1).T
    X_ar2 = state_from_tfms_no_velocity(Ts_ar2).T
    X_hy  = state_from_tfms_no_velocity(Ts_hy).T

    plot_kalman(X_kf[:,1:], X_ks[:,1:], X_ar1, vs_ar1, X_ar2, vs_ar2, X_hy, vs_hy)
    plt.show()
