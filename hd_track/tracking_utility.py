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
import scipy.linalg as scl, scipy.interpolate as si
import math
import matplotlib.pylab as plt

from hd_utils.colorize import colorize
from hd_utils.conversions import *
from hd_utils.utils import *
from hd_utils.defaults import tfm_link_rof


from hd_track.kalman_jia import kalman, closer_angle
from hd_track.kalman import smoother


from hd_track.kalman_tuning import state_from_tfms_no_velocity
from hd_track.streamer import streamize
from hd_track.stream_pc import streamize_pc, streamize_rgbd_pc
from hd_visualization.ros_vis import draw_trajectory 

import hd_utils.transformations as tfms
from hd_utils.defaults import calib_files_dir

hd_path = os.getenv('HD_DIR')

def load_covariances():
    """
    Load the noise covariance matrices:
    """
    covar_mats   =  cp.load(open(hd_path + '/hd_track/data/nodup-covars-1.cpickle'))
    
    
    ar_covar     =  1e2*covar_mats['kinect']
    motion_covar =  1e-3*covar_mats['process']
    hydra_covar  =  covar_mats['hydra']

    motion_covar = np.diag(np.diag(motion_covar))
    ar_covar     = np.diag(np.diag(ar_covar))
    hydra_covar  = np.diag(np.diag(hydra_covar))
    
    #make motion covariance large
    motion_covar = 1e-3*np.eye(12) # motion covar 1e-3
    #print ar_covar
    #print hydra_covar
    #print motion_covar
    
    hydra_covar = 1e-4*np.eye(6) # for rpy 1e-4 
    hydra_covar[0:3,0:3] = 1e-2*np.eye(3) # for xyz 1e-2
    
    hydra_vcovar = 1e-3*np.eye(6) # for xyz-v 1e-5
                                   # for rpy-v 1e-5

    return (motion_covar, ar_covar, hydra_covar, hydra_vcovar)

def load_data(fname, lr, single_camera):
    """
    return cam1, (cam2), hydra transform data.
    """
    if lr == None:
        dat = cp.load(open(fname))
    else:
        dat = cp.load(open(fname))[lr]
        
    cam1_ts  = np.array([tt[1] for tt in dat['camera1']])  ## time-stamps
    cam1_tfs = [tt[0] for tt in dat['camera1']]  ## transforms
    
    cam2_ts = np.array([])
    cam2_tfs = []
    if not single_camera:
        cam2_ts  = np.array([tt[1] for tt in dat['camera2']])  ## time-stamps
        cam2_tfs = [tt[0] for tt in dat['camera2']]  ## transforms

    hydra_ts  = np.array([tt[1] for tt in dat['hydra']])  ## time-stamps
    hydra_tfs = [tt[0] for tt in dat['hydra']]  ## transforms
    
    return (cam1_ts, cam1_tfs, cam2_ts, cam2_tfs, hydra_ts, hydra_tfs)


def relative_time_streams(fname, lr, freq, single_camera):
    '''
    return start-end time, number of time samples, and streams for each sensor
    '''
    c1_ts, c1_tfs, c2_ts, c2_tfs, hy_ts, hy_tfs = load_data(fname, lr, single_camera)
    dt =1./freq

    if c2_ts.any():
        tmin = min(np.min(c1_ts), np.min(c2_ts), np.min(hy_ts))
        tmax = max(np.max(c1_ts), np.max(c2_ts), np.max(hy_ts))
    else:
        tmin = min(np.min(c1_ts), np.min(hy_ts))
        tmax = max(np.max(c1_ts), np.max(hy_ts))

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


def setup_kalman(fname, lr, freq, single_camera):
    c1_ts, c1_tfs, c2_ts, c2_tfs, hy_ts, hy_tfs = load_data(fname, lr, single_camera)
    motion_var, ar_var, hydra_var, hydra_vvar = load_covariances()


    dt = 1./freq
    
    if c2_ts.any():
        tmin = min(np.min(c1_ts), np.min(c2_ts), np.min(hy_ts))
        tmax = max(np.max(c1_ts), np.max(c2_ts), np.max(hy_ts))
    else:
        tmin = min(np.min(c1_ts), np.min(hy_ts))
        tmax = max(np.max(c1_ts), np.max(hy_ts))     
    
    ## calculate the number of time-steps for the kalman filter.    
    nsteps = int(math.ceil((tmax-tmin)/dt))
    
    ## get rid of absolute time, put the three data-streams on the same time-scale
    c1_ts -= tmin
    c2_ts -= tmin
    hy_ts -= tmin
    
    # initialize KF:
    x0, S0 = get_first_state(dt, c1_ts, c1_tfs, c2_ts, c2_tfs, hy_ts, hy_tfs)
    KF = kalman()
    
    ## ===> assumes that the variance for ar1 and ar2 are the same!!!    
    KF.init_filter(-dt, x0, S0, motion_var, hydra_var, hydra_vvar, ar_var, ar_var)
    
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
    except:
        pass
    return ret

def fit_spline_to_stream(strm, nsteps, deg=3):
    x = []
    y = []

    prev_rpy = None
    for i in xrange(nsteps):
        next = soft_next(strm)
        if next is not None:
            pos = next[0:3,3]
            rpy = np.array(tfms.euler_from_matrix(next), ndmin=2).T
            rpy = np.squeeze(rpy)
            if prev_rpy is not None:
                rpy = closer_angle(rpy, prev_rpy)
            prev_rpy = rpy

            x.append(i)
            y.append(pos.tolist()+rpy.tolist())

    x = np.asarray(x)
    y = np.asarray(y)

    s = len(x)*.001**2
    (tck, _) = si.splprep(y.T, s=s, u=x, k=deg)
    
    new_x = xrange(nsteps)
    xyzrpys = np.r_[si.splev(new_x, tck)].T
    
    smooth_tfms = []
    for xyzrpy in xyzrpys:
        tfm = tfms.euler_matrix(*xyzrpy[3:6])
        tfm[0:3,3] = xyzrpy[0:3]
        smooth_tfms.append(tfm)
        
    return smooth_tfms

def run_kalman_filter(fname, lr, freq, use_spline=False, use_hydra=True, single_camera=False):
    """
    Runs the kalman filter
    """
    dt = 1/freq
    KF, nsteps, tmin, ar1_strm, ar2_strm, hy_strm = setup_kalman(fname, lr, freq, single_camera)
    
    ## run the filter:
    mu,S = [], []
    
    if use_hydra and use_spline:
        smooth_hy = (t for t in fit_spline_to_stream(hy_strm, nsteps))
    else:
        smooth_hy = hy_strm
    

    for i in xrange(nsteps):
        if use_hydra:
            KF.register_observation(dt*i, soft_next(ar1_strm), soft_next(ar2_strm), soft_next(smooth_hy)) 
        else:
            KF.register_observation(dt*i, soft_next(ar1_strm), soft_next(ar2_strm), None)
           
        mu.append(KF.x_filt)
        S.append(KF.S_filt)

    A, R = KF.get_motion_mats(dt)
    
    return nsteps, tmin, mu, S, A, R


def get_cam_transform(calib_fname):
    calib_file_fullname = osp.join(calib_files_dir, calib_fname)
    dat = cp.load(open(calib_file_fullname))
    
    camera_tf = None;
    for tf in dat['transforms']:
        if tf['parent'] == 'camera1_link' and tf['child'] == 'camera2_link':
            camera_tf = tf['tfm']
    
    if camera_tf != None:
        T_l1l2 = camera_tf
        return np.linalg.inv(tfm_link_rof).dot(T_l1l2.dot(tfm_link_rof))
    else:
        return np.eye(4)

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


def open_frac(angle):
    """
    Convert the angle in degrees to a fraction.
    """
    angle_max = 33.0
    return angle/angle_max


def plot_kalman_core(X_kf, X_ks, X_ar1, vs_ar1, X_ar2, vs_ar2, X_hy, vs_hy, plot_commands):
    """
    X_kf: kalman filter result
    X_ks: kalman smoother result
    [vs_ar1, X_ar1]: the timing and transform of Ar marker1
    [vs_ar2, X_ar2]: the timing and transform of Ar marker2
    [vs_hy, X_hy]  : the timing and transform of hydra
    """
    if plot_commands == '': return

    to_plot= [i for i in xrange(9)]
    axlabels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw', 'v_roll', 'v_pitch', 'v_yaw']
    for i in to_plot:
        plt.subplot(3,3,i+1)
        if 'f' in plot_commands:
            plt.plot(X_kf[i,:], label='filter')
        if 's' in plot_commands:
            plt.plot(X_ks[i,:], label='smoother')
        if '1' in plot_commands:
            plt.plot(vs_ar1, X_ar1[i,:], '.', label='camera1')
        if '2' in plot_commands:
            plt.plot(vs_ar2, X_ar2[i,:], '.', label='camera2')
        if 'h' in plot_commands:
            plt.plot(vs_hy, X_hy[i,:], '.', label='hydra')
        plt.ylabel(axlabels[i])


def correlation_shift(xa,xb):
    shifts = []
    for idx in [0, 1, 2]:
        shifts.append(np.argmax(np.correlate(xa[idx,:], xb[idx,:], 'full'))-(xb.shape[1]-1))
    print shifts
    return  int(np.max(shifts))


def plot_kalman_lr(data_file, calib_file, lr, freq, use_spline, customized_shift, single_camera, plot_commands):
    '''
    input: data_file
    lr: 'l' or 'r'
    freq
    use_spline
    customized_shift: custimized shift between smoother and filter: to compensate for the lag of smoother
    '''
    _, _, _, ar1_strm, ar2_strm, hy_strm = relative_time_streams(data_file, lr, freq, single_camera)    
    
    ## run kalman filter:
    nsteps, tmin, F_means,S,A,R = run_kalman_filter(data_file, lr, freq, use_spline, True, single_camera)
    ## run kalman smoother:
    S_means, _ = smoother(A, R, F_means, S)
    
    
    X_kf = np.array(F_means)
    X_kf = np.reshape(X_kf, (X_kf.shape[0], X_kf.shape[1])).T

    X_ks = np.array(S_means)
    X_ks = np.reshape(X_ks, (X_ks.shape[0], X_ks.shape[1])).T
    

    # Shifting
    if customized_shift != None:
        shift = customized_shift
    else:
        shift = correlation_shift(X_kf, X_ks)
        

    X_ks = np.roll(X_ks,shift,axis=1)
    X_ks[:,:shift]  = X_ks[:,shift][:,None]

  
    ## frame of the filter estimate:
    indices_ar1 = []
    indices_ar2 = []
    indices_hy  = []

    Ts_ar1 = []
    Ts_ar2 = []
    Ts_hy  = []

    if use_spline:
        smooth_hy = (t for t in fit_spline_to_stream(hy_strm, nsteps))
    else:
        smooth_hy = hy_strm

    for i in xrange(nsteps):
        ar1_est = soft_next(ar1_strm)
        ar2_est = soft_next(ar2_strm)
        hy_est = soft_next(smooth_hy)
        
        if ar1_est != None:
            Ts_ar1.append(ar1_est)
            indices_ar1.append(i)
        if ar2_est != None:
            Ts_ar2.append(ar2_est)
            indices_ar2.append(i)
        if hy_est != None:
            Ts_hy.append(hy_est)
            indices_hy.append(i)

    X_ar1 = state_from_tfms_no_velocity(Ts_ar1).T
    X_ar2 = state_from_tfms_no_velocity(Ts_ar2).T
    X_hy  = state_from_tfms_no_velocity(Ts_hy).T

    plot_kalman_core(X_kf[:,1:], X_ks[:,1:], X_ar1, indices_ar1, X_ar2, indices_ar2, X_hy, indices_hy, plot_commands)
    plt.show()
    
    
def plot_kalman(data_file, calib_file, freq, use_spline=False, customized_shift=None, single_camera=False, plot_commands='s12fh'):
    dat = cp.load(open(data_file))
    if dat.has_key('l'):
        plot_kalman_lr(data_file, calib_file, 'l', freq, use_spline, customized_shift, single_camera, plot_commands)
    if dat.has_key('r'):
        plot_kalman_lr(data_file, calib_file, 'r', freq, use_spline, customized_shift, single_camera, plot_commands)
    
    
    
def rviz_kalman(demo_dir, bag_file, data_file, calib_file, freq, use_rgbd, use_smoother, use_spline, customized_shift, single_camera):
    '''
    For rgbd, data_file and bag_file are redundant
    Otherwise, demo_dir is redundant
    '''
    if use_rgbd:
        bag_file  = osp.join(demo_dir, 'demo.bag')
        rgbd1_dir = osp.join(demo_dir, 'camera_#1')
        rgbd2_dir = osp.join(demo_dir, 'camera_#2')
        data_file = osp.join(demo_dir, 'demo.data') 
        bag = rosbag.Bag(bag_file)
    else:
        bag = rosbag.Bag(bag_file)

    dat = cp.load(open(data_file))
    grippers = dat.keys()

    pub = rospy.Publisher('/point_cloud1', PointCloud2)
    pub2= rospy.Publisher('/point_cloud2', PointCloud2)
    
    c1_tfm_pub = {}
    c2_tfm_pub = {}
    hydra_tfm_pub = {}
    T_filt = {}
    ang_strm = {}
    ar1_strm = {} 
    ar2_strm = {}
    hy_strm = {}
    smooth_hy = {}
    
    ## publishers for unfiltered-data:
    for lr in grippers:
        c1_tfm_pub[lr] = rospy.Publisher('/%s_ar1_estimate'%(lr), PoseStamped)
        c2_tfm_pub[lr] = rospy.Publisher('/%s_ar2_estimate'%(lr), PoseStamped)
        hydra_tfm_pub[lr] = rospy.Publisher('/%s_hydra_estimate'%(lr), PoseStamped)
        
        _, _, _, ar1_strm[lr], ar2_strm[lr], hy_strm = relative_time_streams(data_file, lr, freq, single_camera)
         
        ## run the kalman filter:
        nsteps, tmin, F_means, S,A,R = run_kalman_filter(data_file, lr, freq, use_spline, True, single_camera)
        
        ## run the kalman smoother:
        S_means, _ = smoother(A, R, F_means, S)
        
        X_kf = np.array(F_means)
        X_kf = np.reshape(X_kf, (X_kf.shape[0], X_kf.shape[1])).T
        
        X_ks = np.array(S_means)
        X_ks = np.reshape(X_ks, (X_ks.shape[0], X_ks.shape[1])).T


        # Shifting between filter and smoother:
        if customized_shift != None:
            shift = customized_shift
        else:
            shift = correlation_shift(X_kf, X_ks)

        X_ks = np.roll(X_ks,shift,axis=1)
        X_ks[:,:shift]  = X_ks[:,shift][:,None]
        
        if use_smoother:
            T_filt[lr] = state_to_hmat(list(X_ks.T))
        else:
            T_filt[lr] = state_to_hmat(list(X_kf.T))
        
        ## load the potentiometer-angle stream:
        pot_data = cp.load(open(data_file))[lr]['pot_angles']
        
        ang_ts       = np.array([tt[1] for tt in pot_data])  ## time-stamps
        ang_vals     = [tt[0] for tt in pot_data]  ## angles
#         plt.plot(ang_vals)
#         plt.show()
        ang_vals = [0*open_frac(x) for x in ang_vals]

        ang_strm[lr] = streamize(ang_vals, ang_ts, freq, lambda x : x[-1], tmin)
        
        if use_spline:
            smooth_hy[lr] = (t for t in fit_spline_to_stream(hy_strm, nsteps))
        else:
            smooth_hy[lr] = hy_strm

    ## get the point-cloud stream
    cam1_frame_id = '/camera1_rgb_optical_frame'
    cam2_frame_id = '/camera2_rgb_optical_frame'
    
    if use_rgbd:
        pc1_strm = streamize_rgbd_pc(rgbd1_dir, cam1_frame_id, freq, tmin)
        if single_camera:
            pc2_strm = streamize_rgbd_pc(None, cam2_frame_id, freq, tmin)
        else:
            pc2_strm = streamize_rgbd_pc(rgbd2_dir, cam2_frame_id, freq, tmin)
    else:
        pc1_strm = streamize_pc(bag, '/camera1/depth_registered/points', freq, tmin)
        if single_camera:
            pc2_strm = streamize_pc(bag, None, freq, tmin) 
        else:
            pc2_strm = streamize_pc(bag, '/camera2/depth_registered/points', freq, tmin)  
            

    ## get the relative-transforms between the cameras:
    cam_tfm  = get_cam_transform(calib_file)
    publish_static_tfm(cam1_frame_id, cam2_frame_id, cam_tfm)

    ## frame of the filter estimate:
    sleeper = rospy.Rate(freq)
    T_far = np.eye(4)
    T_far[0:3,3] = [10,10,10]        
    
    handles = []
    
    prev_ang = {'l': 0, 'r': 0}
    for i in xrange(nsteps):
        #raw_input("Hit next when ready.")
        print "Kalman ts: ", tmin+(0.0+i)/freq
        
        ## show the point-cloud:
        found_pc = False
        try:
            pc = pc1_strm.next()
            if pc is not None:
                print "pc1 ts:", pc.header.stamp.to_sec()
                pc.header.stamp = rospy.Time.now()
                pub.publish(pc)
                found_pc = True
            else:
                print "pc1 ts:",None
        except StopIteration:
            print "pc1 ts: finished"
            #print "no more point-clouds"
            pass

        try:
            pc2 = pc2_strm.next()
            if pc2 is not None:
                #print "pc2 not none"
                print "pc2 ts:", pc2.header.stamp.to_sec()
                pc2.header.stamp = rospy.Time.now()
                pub2.publish(pc2)
                found_pc = True
            else:
                print "pc2 ts:", None
        except StopIteration:
            print "pc2 ts: finished"
            pass


        ang_vals  = []
        T_filt_lr = []
        for lr in grippers:
            ang_val = soft_next(ang_strm[lr])
            if ang_val != None:
                prev_ang[lr] = ang_val
                ang_val  = ang_val
            else:
                ang_val = prev_ang[lr]

            ang_vals.append(ang_val)
            T_filt_lr.append(T_filt[lr][i])

        handles = draw_trajectory(cam1_frame_id, T_filt_lr, color=(1,1,0,1), open_fracs=ang_vals)

        # draw un-filtered estimates:
        for lr in grippers:
            ar1_est = soft_next(ar1_strm[lr])
            if ar1_est != None:
                c1_tfm_pub[lr].publish(pose_to_stamped_pose(hmat_to_pose(ar1_est), cam1_frame_id))
            else:
                c1_tfm_pub[lr].publish(pose_to_stamped_pose(hmat_to_pose(T_far), cam1_frame_id))
                
            ar2_est = soft_next(ar2_strm[lr])
            if ar2_est != None:
                c2_tfm_pub[lr].publish(pose_to_stamped_pose(hmat_to_pose(ar2_est), cam1_frame_id))
            else:
                c2_tfm_pub[lr].publish(pose_to_stamped_pose(hmat_to_pose(T_far), cam1_frame_id))
            
            hy_est = soft_next(smooth_hy[lr])
            if hy_est != None:
                hydra_tfm_pub[lr].publish(pose_to_stamped_pose(hmat_to_pose(hy_est), cam1_frame_id))
            else:
                hydra_tfm_pub[lr].publish(pose_to_stamped_pose(hmat_to_pose(T_far), cam1_frame_id))
        
        sleeper.sleep()
        

        
def traj_kalman_lr(data_file, calib_file, lr, freq, use_spline, customized_shift, single_camera):
    '''
    input: data_file
    lr: 'l' or 'r'
    freq
    use_spline
    customized_shift: custimized shift between smoother and filter: to compensate for the lag of smoother
    
    return trajectory data
    '''
    
    _, _, _, ar1_strm, ar2_strm, hy_strm = relative_time_streams(data_file, lr, freq, single_camera)    
    
    ## run kalman filter:
    nsteps, tmin, F_means, S, A, R = run_kalman_filter(data_file, lr, freq, use_spline, True, single_camera)
    ## run kalman smoother:
    S_means, _ = smoother(A, R, F_means, S)
    
    
    X_kf = np.array(F_means)
    X_kf = np.reshape(X_kf, (X_kf.shape[0], X_kf.shape[1])).T

    X_ks = np.array(S_means)
    X_ks = np.reshape(X_ks, (X_ks.shape[0], X_ks.shape[1])).T
    

    # Shifting
    if customized_shift != None:
        shift = customized_shift
    else:
        shift = correlation_shift(X_kf, X_ks)
        

    X_ks = np.roll(X_ks,shift,axis=1)
    X_ks[:,:shift]  = X_ks[:,shift][:,None]
    
    T_filter = state_to_hmat(list(X_ks.T))
    T_smoother = state_to_hmat(list(X_kf.T))
    
    ## load the potentiometer-angle stream:
    pot_data = cp.load(open(data_file))[lr]['pot_angles']
    
    ang_ts   = np.array([tt[1] for tt in pot_data])  ## time-stamps
    ang_vals = [open_frac(tt[0]) for tt in pot_data]  ## angles
    ang_strm = streamize(ang_vals, ang_ts, freq, lambda x : x[-1], tmin)
    
    
    ang_strm_vals = []

    prev_ang = 0
    for i in xrange(nsteps):
        ang_val = soft_next(ang_strm)
        if ang_val != None:
            prev_ang = ang_val
            ang_val  = [ang_val]
        else:
            ang_val = [prev_ang]
        
        ang_strm_vals.append(ang_val)
        
    stamps = []
    for i in xrange(nsteps):
        stamps.append(tmin + i * 1.0 / freq)
    
        
    traj = {"tfms": T_filter, "tfms_s": T_smoother, "pot_angles": ang_strm_vals, "stamps": stamps}
    
    return traj
    
def traj_kalman(data_file, calib_file, freq, use_spline=False, customized_shift=None, single_camera=False):
    traj = {}
    data = cp.load(open(data_file))

    if data.has_key('l'):
        traj['l'] = traj_kalman_lr(data_file, calib_file, 'l', freq, use_spline, customized_shift, single_camera)
    if data.has_key('r'):
        traj['r'] = traj_kalman_lr(data_file, calib_file, 'r', freq, use_spline, customized_shift, single_camera)

    return traj

    


