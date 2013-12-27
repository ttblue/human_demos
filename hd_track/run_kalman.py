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

from hd_utils.colorize import colorize, redprint, blueprint
from hd_utils.conversions import *
from hd_utils.utils import *
from hd_utils.defaults import tfm_link_rof

from hd_track.kalman import kalman, closer_angle
from hd_track.kalman import smoother
from hd_track.kalman_tuning import state_from_tfms_no_velocity

from hd_track.streamer import streamize, soft_next, get_corresponding_data
from hd_track.stream_pc import streamize_pc, streamize_rgbd_pc
from hd_visualization.ros_vis import draw_trajectory 

from hd_utils.defaults import calib_files_dir, hd_path

from hd_track.demo_data_prep import *
from hd_track.tps_correct    import *


def initialize_covariances(freq=30.0):
    """
    Initialize empirical estimates of covariances:
    
     -- Cameras and the hydra observe just the xyzrpy (no velocities).
     -- Motion covariance is for all 12 variables.
    """
    cam_xyz_std    = 0.01 # 1cm
    hydra_xyz_std  = 0.02 # 2cm <-- use small std after tps-correction.
    motion_xyz_std = 0.05 # 5cm <-- motion model is a hack => use large covariance.
    
    cam_rpy_std    = np.deg2rad(15)
    hydra_rpy_std  = np.deg2rad(5)
    motion_rpy_std = np.deg2rad(20)
    
    ## initialize velocity covariances: divide by the filter frequency to put on the correct time-scale
    motion_vx_std  = 0.05/freq # 5cm/s 
    motion_vth_std = np.deg2rad(10/freq) # 10 deg/s

    I3 = np.eye(3)
    cam_covar    =  scl.block_diag(   np.square(cam_xyz_std)*I3, np.square(cam_rpy_std)*I3  )
    hydra_covar  =  scl.block_diag(   np.square(hydra_xyz_std)*I3, np.square(hydra_rpy_std)*I3  )
    motion_covar =  scl.block_diag(   np.square(motion_xyz_std)*I3,
                                      np.square(motion_vx_std)*I3,
                                      np.square(motion_rpy_std)*I3,
                                      np.square(motion_vth_std)*I3  )

    return (motion_covar, cam_covar, hydra_covar)


def get_first_state(tf_streams, freq=30.):
    """
    Returns the first state and covariance given :
     - TF_STREAMS : a list of streams of transform data.
     - FREQ       : the frequency of these streams (and also the filter).
    """
    dt = 1./freq
    n_streams = len(tf_streams)

    tfs0 = []
    for i in xrange(n_streams):
        tfs, ts = tf_streams[i].get_data()
        for ti, t in enumerate(ts):
            if t <= dt:
                tfs0.append(tfs[ti])

    if len(tfs0)==0:
        redprint("Cannot find initial state for KF: no data found within dT in all streams")    

    x0 =  state_from_tfms_no_velocity([avg_transform(tfs0)])
    I3 = np.eye(3)
    S0 = scl.block_diag(1e-3*I3, 1e-2*I3, 1e-3*I3, 1e-3*I3)
    return (x0, S0)


def plot_tf_streams(tf_strms, strm_labels, block=True):
    """
    Plots the x,y,z,r,p,y from a list TF_STRMS of streams.
    """
    assert len(tf_strms)==len(strm_labels)
    ylabels = ['x', 'y', 'z', 'r', 'p', 'y']
    n_streams = len(tf_strms)
    Xs   = []
    inds = []
    for strm in tf_strms:
        tfs, ind = [], []
        for i,tf in enumerate(strm):
            if tf != None:
                tfs.append(tfs)
                ind.append(i)
        X = state_from_tfms_no_velocity(tfs, 6)
        Xs.append(X)
        inds.append(ind)
    
    for i in xrange(6):
        plt.subplot(2,3,i+1)
        plt.hold(True)
        for j in xrange(n_streams):
            xj = Xs[j]
            ind_j = inds[j]
            plt.plot(ind_j, xj[:,i], label=strm_labels[j])
            plt.ylabel(ylabels[i])
        plt.legend()
    plt.show(block=block)


def load_data_for_kf(dat_fname, lr, freq=30., hy_tps_fname=None, plot=False):
    """
    Collects all the data in the correct format/ syncs it to run on the kalman filter.
    
    DAT_FNAME   : File name of the 'demo.data' as saved by extract_data
    LR          : {'r', 'l'} : right/ left gripper 
    HY_TPS_FNAME: File name of the saved tps-model to use. 
                  If None this function fits a tps model based on the current data file.
    """
    tfm_data, pot_data, T_cam2hbase, T_tt2hy = load_data(dat_fname, lr)
    _,_,_, tf_streams = relative_time_streams(tfm_data, freq)

    hy_strm   = tf_streams[0]
    hy_strm   = fit_spline_to_tf_stream(hy_strm, freq)
    cam_strms = tf_streams[1:]

    strm_labels =  ['hydra'] + ['cam%d'%(i+1) for i in xrange(len(cam_strms))]
    if plot:
        blueprint("Plotting raw (unaligned) data-streams...")
        plot_tf_streams([hy_strm]+cam_strms, strm_labels, block=False)


    ## time-align all the transform streams:
    tmin, tmax, nsteps, hy_strm, cam_strms = align_all_streams(hy_strm, cam_strms)

    if plot:
        blueprint("Plotting ALIGNED data-streams...")
        plot_tf_streams([hy_strm]+cam_strms, strm_labels, block=True)

    ### TPS correct the hydra data:
    if hy_tps_fname==None:
        # train a tps-model:
        ## NOTE THE TPS-MODEL is fit b/w hydra and CAMERA'1'
        _, hy_tfs, cam1_tfs = get_corresponding_data(hy_strm, cam_strms[0])
        n_matching   = len(hy_tfs)
        x_hy, x_cam  = np.empty((n_matching,3)), np.empty((n_matching,3))
        for i in xrange(n_matching):
            x_hy[i,:]  = hy_tfs[i][0:3,3]
            x_cam[i,:] = cam1_tfs[i][0:3,3]
        f_tps = fit_tps(x_cam, x_hy, plot)
    else:
        f_tps = load_tps(hy_tps_fname)

    hy_tfs, hy_ts  = hy_strm.get_data()
    hy_tfs_aligned = correct_hydra(hy_tfs, T_tt2hy, T_cam2hbase, f_tps) 
    hy_corr_strm   = streamize(hy_tfs_aligned, hy_ts, hy_strm.freq, hy_strm.favg, hy_strm.tstart)

    if plot:
        blueprint("Plotting tps-corrected hydra...")
        plot_tf_streams([hy_strm, hy_corr_strm, cam_strms[0]], ['hy', 'hy-tps', 'cam1'])

    ## now setup and run the kalman-filter:
    motion_covar, cam_covar, hydra_covar = initialize_covariances(freq)
    x0, S0 = get_first_state([hy_corr_strm]+cam_strms, freq)
    KF = kalman()
    KF.init_filter(0, x0, S0, motion_covar, cam_covar, hydra_covar)




def run_kalman_filter(fname, lr, freq, use_spline=False, use_hydra=True, single_camera=False):
    """
    Runs the kalman filter
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
    """
    pass



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


def plot_kalman_lr(data_file, calib_file, lr, freq, use_spline, customized_shift, single_camera, plot_commands):
    '''
    input: data_file
    lr: 'l' or 'r'
    freq
    use_spline
    customized_shift: custimized shift between smoother and filter: to compensate for the lag of smoother
    
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
    '''
    pass
    
def plot_kalman(data_file, calib_file, freq, use_spline=False, customized_shift=None, single_camera=False, plot_commands='s12fh'):
    dat = cp.load(open(data_file))
    for lr in 'lr':
        if dat.has_key(lr):
            plot_kalman_lr(data_file, calib_file, lr, freq, use_spline, customized_shift, single_camera, plot_commands)

    
def rviz_kalman(demo_dir, bag_file, data_file, calib_file, freq, use_rgbd, use_smoother, use_spline, customized_shift, single_camera):
    '''
    For rgbd, data_file and bag_file are redundant
    Otherwise, demo_dir is redundant
    
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
        '''
    pass
        

        
def traj_kalman_lr(data_file, calib_file, lr, freq, use_spline, customized_shift, single_camera):
    '''
    input: data_file
    lr: 'l' or 'r'
    freq
    use_spline
    customized_shift: custimized shift between smoother and filter: to compensate for the lag of smoother
    
    return trajectory data
    '''
    pass
    #traj = {"tfms": T_filter, "tfms_s": T_smoother, "pot_angles": ang_strm_vals, "stamps": stamps}
    #return traj
    

def traj_kalman(data_file, calib_file, freq, use_spline=False, customized_shift=None, single_camera=False):
    traj = {}
    data = cp.load(open(data_file))
    if data.has_key('pot_angles'):
        traj['l'] = traj_kalman_lr(data_file, calib_file, None, freq, use_spline, customized_shift, single_camera)
    else:
        if data.has_key('l'):
            traj['l'] = traj_kalman_lr(data_file, calib_file, 'l', freq, use_spline, customized_shift, single_camera)
        if data.has_key('r'):
            traj['r'] = traj_kalman_lr(data_file, calib_file, 'r', freq, use_spline, customized_shift, single_camera)

    return traj

    


