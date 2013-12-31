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


def plot_tf_streams(tf_strms, strm_labels, styles=None, title=None, block=True):
    """
    Plots the x,y,z,r,p,y from a list TF_STRMS of streams.
    """
    assert len(tf_strms)==len(strm_labels)
    if styles!=None:
        assert len(tf_strms)==len(styles)
    else:
        styles = ['-']*len(tf_strms)

    plt.figure()
    ylabels = ['x', 'y', 'z', 'r', 'p', 'y']
    n_streams = len(tf_strms)
    Xs   = []
    inds = []
    for strm in tf_strms:
        tfs, ind = [], []
        for i,tf in enumerate(strm):
            if tf != None:
                tfs.append(tf)
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
            plt.plot(ind_j, xj[:,i], styles[j], label=strm_labels[j])
            plt.ylabel(ylabels[i])
        plt.legend()

    if title!=None:
        plt.gcf().suptitle(title)
    plt.show(block=block)



def load_data_for_kf(demo_fname, freq=30.0, rem_outliers=True, tps_correct=True, tps_model_fname=None, plot=False):
    cam_dat = {}
    hy_dat  = {}
    pot_dat = {}
    dt      = 1./freq
    demo_dir  = osp.dirname(demo_fname)
    lr_full   = {'l': 'left', 'r':'right'}

    T_cam2hbase, cam_dat['l'], hy_dat['l'], pot_dat['l'] = load_data(demo_fname, 'l', freq)
    T_cam2hbase, cam_dat['r'], hy_dat['r'], pot_dat['r'] = load_data(demo_fname, 'r', freq)
    
    ## collect all camera streams in a list:
    all_cam_names = set()
    for lr in 'lr':
        for cam in cam_dat[lr].keys():
            all_cam_names.add(cam)

    ## remove outliers in the camera streams
    if rem_outliers:
        blueprint("Rejecting TF outliers in camera-data..")
        for lr in 'lr':
            for cam in cam_dat[lr].keys():
                strm_in = reject_outliers_tf_stream(cam_dat[lr][cam]['stream'])
                if plot:
                    cam_name = cam+'_'+lr
                    plot_tf_streams([cam_dat[lr][cam]['stream'], strm_in], strm_labels=[cam_name, cam_name+'_in'], styles=['.','-'], block=False)
                cam_dat[lr][cam]['stream'] = strm_in


    ## time-align all tf-streams (wrt their respective hydra-streams):
    ## NOTE THE STREAMS ARE MODIFIED IN PLACE (the time-stamps are changed)
    ## and also the tstart
    blueprint("Getting rid of absolute time..")
    all_cam_strms = []
    for lr in 'lr':
        for cam in cam_dat[lr].keys():
            all_cam_strms.append(cam_dat[lr][cam]['stream'])
    tmin, _, _ = relative_time_streams(hy_dat.values() + pot_dat.values() + all_cam_strms)
    tshift1 = -tmin

    ## do time-alignment (with the respective l/r hydra-streams):
    blueprint("Calculating TF stream time-shifts..")
    time_shifts = {}   ## dictionary : maps a camera-name to the time-shift wrt hydra
    for cam in all_cam_names:
        ndat = {'r':-1, 'l':-1}  ## compare whether left/right has more data-points
        for lr in 'lr':
            if cam in cam_dat[lr].keys():
                ndat[lr] = len(cam_info[lr][cam]['stream'].get_data()[0])

        lr_align = ndat.keys[np.argmax(ndat.values())]
        time_shifts[cam] = dt* align_tf_streams(hy_dat[lr_align], cam_dat[lr_align][cam]['stream'])

    redprint("\t Time-shifts found : ")
    print time_shifts
    
    ## time-shift the streams:
    blueprint("Time-aligning TF streams..")
    for lr in 'lr':
        aligned_cam_strms = []
        for cam in cam_dat[lr].keys():
            aligned_cam_strms.append( time_shift_stream(cam_info[lr][cam]['stream'], time_shifts[cam]) )
                
        if plot:
            unaligned_cam_streams = []
            for cam_dat in cam_dat[lr].values():
                unaligned_cam_streams.append(cam_dat['stream'])
            plot_tf_streams(unaligned_cam_streams + [hy_dat[lr]], cam_dat[lr].keys()+['hydra'], title='UNALIGNED CAMERA-streams (%s)'%lr_full[lr], block=False)
            plot_tf_streams(aligned_cam_strms+hy_dat[lr], cam_dat[lr].keys()+['hydra'], title='ALIGNED CAMERA-streams (%s)'%lr_full[lr], block=False)
            
        for i,cam in enumerate(cam_dat[lr].keys()):
            cam_dat[lr][cam]['stream'] = aligned_cam_strms[i]


    ## put the aligned-streams again on the same time-scale: 
    blueprint("Re-aligning the TF-streams after TF based time-shifts..")
    tmin, tmax, nsteps = relative_time_streams(hy_dat.values() + pot_dat.values() + all_cam_strms)
    tshift2 = -tmin

    ## TPS-correct the hydra-data:
    if tps_correct:
        if tps_model_fname == None:
            blueprint("\t Fitting TPS model to demo data..")
            tps_models = {}
            for lr in 'lr':
                _, hy_tfs, cam1_tfs = get_corresponding_data(hy_dat[lr], cam_dat[lr]['camera1']['stream'])
                f_tps = fit_tps_on_tf_data(hy_tfs, cam1_tfs, plot=False) ## <<< mayavi plotter has some issues with multiple insantiations..
                T_cam2hbase_train = T_cam2hbase
                tps_models[lr]    = f_tps

                tps_save_fname = osp.join(demo_dir, 'tps_models.cp')
                with open(tps_save_fname, 'w') as f:
                    cp.dump([tps_models, T_cam2hbase], f)
        else:
            blueprint("\t Getting TPS-model from : %s"%tps_model_fname)
            with open(tps_model_fname, 'r') as f:
                tps_models, T_cam2hbase_train = cp.load(f)


        blueprint("TPS-correcting hydra-data..")
        for lr in 'lr':
            hy_tfs, hy_ts  = hy_dat[lr].get_data()
            hy_tfs_aligned = correct_hydra(hy_tfs, T_cam2hbase, tps_models[lr], T_cam2hbase_train)
            hy_strm_aligned = streamize(hy_tfs_aligned, hy_ts, 1./hy_strm.dt, hy_strm.favg, hy_strm.tstart)
    
            if plot:
                if tps_mode_fname!=None:
                    plot_tf_strms([hy_dat[lr], hy_strm_aligned], ['hy-old', 'hy-corr'], title='hydra-correction %s'%lr_full[lr], block=False)
                else:
                    plot_tf_strms([hy_dat[lr], hy_strm_aligned, cam_dat[lr]['camera1']['stream']], ['hy-old', 'hy-corr', 'cam1'], title='hydra-correction', block=False)

            hy_dat[lr] = hy_strm_aligned


    ## now setup the kalman-filter:
    blueprint("Initializing Kalman filter..")
    motion_covar, cam_covar, hydra_covar = initialize_covariances(freq)
    KFs = {}
    for lr in 'lr':
        blueprint("\t setting up %s kalman-filter"%{'r': 'right', 'l':'left'}[lr])
        blueprint("\t\t Getting starting state for KF..")
        cam_strms = [cdat['stream'] for cdat in cam_dat[lr].values()]
        x0, S0 = get_first_state(cam_strms+[hy_dat[lr]], freq)
        KF = kalman()
        kf_tstart = hy_dat[lr].get_start_time() - dt
        KF.init_filter(kf_tstart, x0, S0, motion_covar, cam_covar, hydra_covar)
        KFs[lr] = KF
        blueprint("\t\t\t Done initializing KF.")

    ## return the data:
    filter_data = {'demo_dir': demo_dir,
                   'KFs'     : KFs,
                   'nsteps' : nsteps,
                   'tmin'   : tmin,
                   'tmax'   : tmax,
                   'pot_dat': pot_dat,
                   'hy_dat': hy_dat,
                   'cam_dat': cam_dat,
                   't_shift': tshift1+tshift2,
                   'cam_shifts': time_shifts}
    return filter_data


def run_kf(filter_data, do_smooth=False, plot=False):
    """
    Actually runs the data-through the kalman filter.
    FILTER_DATA : a dictionary containing all the required data, as returned from load_data_for_kf
    DO_SMOOTH   : if true, the smoother is also run on the data.
    
    Returns : A list of kalman filter state-estimates, covariances and time-stamps.

    Note :
    1. The kalman-filter in filter_data as returned by load_data_for_kf 
       has appropriately been initialized.
    2. Note: This runs the k.f. for only one-gripper.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    NEEDS TO BE FIXED TO USE DIFFERENT COVARIANCES BASED ON CAMERA-TYPES
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    """
    kf_estimates = {}
    cam_types    = get_cam_types(filter_data['demo_dir'])

    for lr in 'lr':
        KF        = filter_data['KF'][lr]
        hy_strm   = filter_data['hy_dat'][lr]
        cam_strms = filter_data['cam_dat'][lr]
        nsteps    = filter_data['nsteps']
        tstart    = hy_strm.get_start_time()

        ## place holders for kalman filter's output:
        xs_kf, covars_kf, ts_kf = [KF.x_filt],[KF.S_filt],[KF.t_filt]
        for i in xrange(nsteps):
            KF.register_tf_observation(soft_next(hy_strm), KF.hydra_covar, do_control_update=True)
            for strm in cam_strms:
                KF.register_tf_observation(soft_next(strm), KF.cam_covar, do_control_update=False)
            xs_kf.append(KF.x_filt)
            covars_kf.append(KF.S_filt)
            ts_kf.append(KF.t_filt)

        if do_smooth:
            A,R  = KF.get_motion_mats()
            xs_smthr, covars_smthr = smoother(A, R, xs_ks, S_kf)
            kf_estimates[lr] = (ts_kf, xs_kf, covars_kf, xs_smthr, covars_smthr)
        else:
            kf_estimates[lr] = (xs_kf, covars_kf, ts_kf)

    return kf_estimates



def filter_traj(demo_fname, mplot=False, rviz=False, tps_model_fname=None, save_tps=False):
    """
    Runs the kalman filter for BOTH the grippers and visualizes the output.
    Also writes the demo.traj file.
    
    MPLOT : Show the data in matplotlib plots
    RVIZ : Visualize the filtered data in rviz
    TPS_MODEL_FNAME : The name of the file to load the tps-model from
    if save_tps:
        demo_dir = osp.dirname(demo_fname)
        save_tps_fname = osp.join(demo_dir, 'tps_model.dat')

    r_data = load_data_for_kf(demo_fname, 'r', freq=30.,
                              fit_spline=False, plot=False,
                              hy_tps_fname=tps_model_fname, 
                              save_tps_fname=save_tps_fname if save_tps else None)


    #traj = {"tfms": T_filter, "tfms_s": T_smoother, "pot_angles": ang_strm_vals, "stamps": stamps}
    """
    pass


def open_frac(angle):
    """
    Convert the angle in degrees to a fraction.
    """
    angle_max = 33.0
    return angle/angle_max


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


if __name__=='__main__':
    load_data_for_kf('../hd_data/demos/demo2_gpr/demo.data', freq=30.0, rem_outliers=True, tps_correct=True, tps_model_fname=None, plot=True)


