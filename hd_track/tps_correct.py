from __future__ import division

import numpy as np
import os, os.path as osp
import cPickle as cp
import pickle
import scipy.linalg as scl, scipy.interpolate as si
import math
import matplotlib.pylab as plt

from hd_utils.colorize import colorize
from hd_utils.conversions import *
from hd_utils.utils import *
from hd_utils.defaults import tfm_link_rof

from hd_track.streamer import streamize, soft_next, time_shift_stream

import hd_utils.transformations as tfms
from   hd_utils.defaults import calib_files_dir
from   hd_utils.defaults import hd_path

## for tps-correct:
from   rapprentice import registration
from   hd_utils.tps_utils import *
from   hd_utils.mayavi_plotter import *


"""
Script to correct hydra-position estimates using Thin-Plate splines.
A TPS model is fit using the data when both the camera and the hydra can
see the marker.

When the camera cannot see the marker, this TPS model is used for interpolation.
"""

def fit_tps(x_gt, x, plot=True):
    """
    Fits a thin-plate spline model to x (source points) and x_gt (ground-truth target points).
    This transform can be used to correct the state-dependent hydra errors.
    """
    bend_coef = 0.00  ## increase this to make the tps-interpolation more smooth
    f = registration.fit_ThinPlateSpline(x, x_gt, bend_coef = bend_coef, rot_coef = 0.001)

    if plot:
        plot_reqs = plot_warping(f.transform_points, x, x_gt, fine=False, draw_plinks=True)
        plotter = PlotterInit()
        for req in plot_reqs:
            plotter.request(req)

    return f.transform_points


def load_data(dat_fname, lr):
    """
    Returns four things for the left/right (based on lr) gripper:
    
    1.  A list of tuples of (transforms, stamps).
        The first tuple is the hydra-data.
        All the rest of the tuples correspond to the n-cameras in the data-file
        (usually 1 or 2).
    2. Potentiometer angles : a tuple of (vals, stamps)
    3. The transform from camera1 to hydra-base.
    4. The transform from tool-tip to the hydra-sensor on the gripper.
    
    Assumes that the data was saved using the save_observations_rgbd function in extract_data.py
    """

    with open(dat_fname, 'r') as f:
        dat = cp.load(f)
    T_cam2hbase = dat['T_cam2hbase']
    T_tt2hy     = dat[lr]['T_tt2hy']

    ## find the number of cameras for which we have data.
    num_cameras = 0
    for kname in dat[lr].keys():
        if 'camera' in kname:
            num_cameras += 1
    
    ## extract transform data:
    tfm_data = []

    ## hydra data:
    hy_tfs = [tt[0] for tt in dat[lr]['hydra']]     
    hy_ts  = np.array([tt[1] for tt in dat[lr]['hydra']])  
    tfm_data.append((hy_tfs, hy_ts))
    
    ## camera data:
    for i in xrange(num_cameras):
        cam_name = 'camera%d'%(i+1)
        cam_tfs = [tt[0] for tt in dat[lr][cam_name]]
        cam_ts  = np.array([tt[1] for tt in dat[lr][cam_name]])
        tfm_data.append((cam_tfs, cam_ts))
        
    ## potentiometer angles:
    pot_vals = np.array([tt[0] for tt in dat[lr]['pot_angles']])
    pot_ts   = np.array([tt[1] for tt in dat[lr]['pot_angles']])

    return (tfm_data, (pot_vals, pot_ts), T_cam2hbase, T_tt2hy)



def relative_time_streams(tfm_data, freq):
    """
    Return start/end time, number of time samples, and streams for each sensor.
    
    - TFM_DATA is a list of tuples of the form : [(tfs_1, ts_1), (tfs_2, ts_2), ...]
        where,
          tfs_i : is a list of transforms
          ts_i  : is a numpy array of the time-stamps corresponding to the transforms.
    
      Note this TFM_DATA is returned as the first parameter from load_data function.
      
    - FREQ : is the frequency at which the data-streams must be streamized.
    """
    
    n_series = len(tfm_data)
    print "Found %d data-streams to streamize."%n_series
    
    dt =1./freq

    ## calculate tmin & tmax to get rid of absolute time scale:
    tmin, tmax = float('inf'), float('-inf')
    for series in tfm_data:
        _, ts = series
        if ts.any():
            tmin = min(tmin, np.min(ts))            
            tmax = max(tmax, np.max(ts)) 

    ## calculate the number of time-steps for the kalman filter.    
    nsteps = int(math.ceil((tmax-tmin)/dt))
    
    ## create the data-streams:
    streams = []
    for series in tfm_data:
        tfs, ts = series
        if ts.any():
            ts -= tmin
            strm = streamize(tfs, ts, freq, avg_transform)

    return tmin, tmax, nsteps, streams



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



def align_tf_streams(hydra_strm, cam_strm, wsize=20):
    """
    Calculates the time-offset b/w the camera and hydra transform streams.
    It uses a hydra-stream because it is full -- it has no missing data-points.
                                                    ===========================

      --> If that is not the case, first fit a spline to the hydra stream and then call this function.
    
    WSIZE : is the window-size to search in : 
            This function gives the:
                argmin_{s \in [-wsize, ..,0,..., wsize]} dist(hy_stream, cam_stream(t+s))
                where, dist is the euclidean norm (l2 norm).

    NOTE : (1) It uses only the position variables for distance calculations.
           (2) Further assumes that the two streams are on the same time-scale.       
    """
    Xs_hy  = []
    Xs_cam = []
    cam_inds = []
    idx = 0
    for tfm in cam_strm:
        if tfm != None:
            Xs_cam.append(tfm[0:3,3])
            cam_inds.append(idx)
        idx += 1

    for hy_tfm in hydra_strm:   
        hy_tfm = hydra_strm.next()
        Xs_hy.append(hy_tfm[0:3,3])

    ## chop-off wsized data from the start and end of the camera-data:
    start_idx, end_idx = 0, len(Xs_cam)-1
    while cam_inds[start_idx] < wsize: start_idx += 1
    while cam_inds[end_idx] >= len(Xs_hy) - wsize: end_idx   -= 1
    
    dists = []
    cam_inds = cam_inds[start_idx:end_idx+1]
    Xs_cam   = np.array(Xs_cam[start_idx:end_idx+1])
    for shift in xrange(-wsize, wsize+1):
        hy_xs = [Xs_hy[idx + shift] for idx in cam_inds]
        dists.append(np.linalg.norm(np.array(hy_xs) - Xs_cam))
    
    return xrange(-wsize, wsize+1)[np.argmin(dists)]



def align_all_streams(hy_strm, cam_streams, wsize=20):
    """
    Time aligns all-streams to camera1 stream.
    
    hy_strm     : The hydra-transforms stream. Assumes that hydra stream is full (no missing data).
    cam_streams : A list of camera-data streams.
                  The first stream is assumed to be camera1 stream.
                  All streams are aligned with camera1 (i.e. camera1 stream is not changed).
    """
    n_streams = len(cam_streams)
    tmin, tmax = float('inf'), float('-inf')
    if n_streams >= 1:
        dt_hydra   = align_tf_streams(hy_strm, cam_streams[0], wsize)
        hy_aligned = time_shift_stream(hy_strm, -dt_hydra)
        
        tmin = min(tmin, np.min(hy_aligned.ts))
        tmax = max(tmax, np.max(hy_aligned.ts))
        
        aligned_streams = []
        for i in xrange(1, n_streams):
            dt = align_tf_streams(hy_strm, cam_streams[i], wsize)
            shifted_stream = time_shift_stream(cam_streams[i], dt-dt_hydra)

            tmin = min(tmin, np.min(shifted_stream.ts))
            tmax = max(tmax, np.max(shifted_stream.ts))

            aligned_streams.append(shifted_stream)

        ## shift the start-times such that all the streams start at the minimum time:
        hy_aligned.tstart -= tmin 
        hy_aligned.t = hy_aligned.tstart
        
        for strm in aligned_streams:
            strm.tstart -= tmin
            strm.t = strm.tstart

        nsteps = (tmax-tmin)/hy_aligned.dt
        return (tmin, tmax, nsteps, hy_aligned, aligned_streams)


  
def get_corresponding_data(plot=False):
    """
    Returns two lists : (1) estimate of the hydra-sensor from hydra-base.
                        (2) estimate of the hydra-sensor from the camera in hydra-bases frame.
    """
    
    ## The hydra and camera data streams are not synced (some delay issues), therefore
    ## first remove that shift using correlation shift.
    _, _, _, ar1_strm, ar2_strm, hy_strm = relative_time_streams(data_file, lr, freq, single_camera)    
    
    ## run the kalman filter on just the camera-data. 
    nsteps, tmin, F_means, S, A,R = run_kalman_filter(data_file, lr, freq, use_spline, False, single_camera)

    X_kf = np.array(F_means)
    X_kf = np.reshape(X_kf, (X_kf.shape[0], X_kf.shape[1])).T

    if use_spline:
        smooth_hy = (t for t in fit_spline_to_stream(hy_strm, nsteps))
    else:
        smooth_hy = hy_strm

    indices_ar1 = []
    indices_ar2 = []
    indices_hy  = []
    Ts_ar1      = []
    Ts_ar2      = []
    Ts_hy       = []

    for i in xrange(nsteps):
        ar1_est = soft_next(ar1_strm)
        ar2_est = soft_next(ar2_strm)
        hy_est  = soft_next(smooth_hy)

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

    ## shift the ar1 data to sync it up with the hydra data:    
    if customized_shift != None:
        shift = customized_shift
    else:
        shift = correlation_shift(X_kf, X_hy)
    
    ## shift the indices of ar1 by the SHIFT calculated above:
    indices_hy = [idx + shift for idx in indices_hy]
    indices_hy = [x for x in indices_hy if 0<= x and x < nsteps]

    Ts_hy_matching  = []
    Ts_ar1_matching = []
    for i,x in enumerate(indices_ar1):
        try:
            idx = indices_hy.index(x)
            Ts_hy_matching.append(Ts_hy[idx])
            Ts_ar1_matching.append(Ts_ar1[i])
        except:
            pass

    print colorize("Found %s corresponding data points b/w camera and hydra." % colorize("%d"%len(Ts_ar1_matching), "red", True), "blue", True)

    if plot:
        X_ar_matching = state_from_tfms_no_velocity(Ts_ar1_matching).T
        X_hy_matching = state_from_tfms_no_velocity(Ts_hy_matching).T    
        plot_kalman_core(X_ar_matching, X_hy_matching, None, None, None, None, None, None, 'fs')
        plt.show()
    
    return (Ts_hy_matching, Ts_ar1_matching)

        
# indices for the training data
def gen_indices(N, k):
    """
    generate indices for splitting data in blocks
    """
    n = int(N/k)
    inds = np.arange(k)
    
    trn = np.empty(0, dtype=int)
    tst = np.empty(0, dtype=int)

    for i in xrange(n):
        if i%2==0:
            trn = np.r_[trn , (k*i + inds)]
        else:
            tst = np.r_[tst , (k*i + inds)]
    return (trn, tst)


def fit_and_plot_tps(plot=False):
    
    T_cam_gt, T_hy = pickle.load(open('grp.dat', 'r'))
    
    T_cam_gt = T_cam_gt[100:len(T_cam_gt)-100]
    T_hy     = T_hy[100:len(T_hy)-100]

    BLOCK_SIZE  = 10
    N           = len(T_hy)
    train, tst  = gen_indices(N, BLOCK_SIZE)

    N_train = len(train)
    N_test  = len(tst)

    T_hy_train     = [T_hy[i] for i in train]
    T_hy_test      = [T_hy[i] for i in tst]
    T_cam_gt_train = [T_cam_gt[i] for i in train]
    T_cam_gt_test  = [T_cam_gt[i] for i in tst]      
    
    x_train    = np.empty((N_train, 3))
    x_test     = np.empty((N_test, 3)) 
    x_gt_train = np.empty((N_train, 3))
    x_gt_test  = np.empty((N_test, 3)) 
    
    for i in xrange(N_train):
        x_train[i,:] = T_hy_train[i][0:3,3]
        x_gt_train[i,:] = T_cam_gt_train[i][0:3,3]
        
    for i in xrange(N_test):
        x_test[i,:] = T_hy_test[i][0:3,3]
        x_gt_test[i,:] = T_cam_gt_test[i][0:3,3]
        
    # fit tps:
    x_gt_train = x_gt_train[100:-100]
    x_train    = x_train[100:-100]
    print colorize("Fitting TPS on %d data-points."%len(x_train), "red", True)
    f_tps = fit_tps(x_gt_train, x_train, plot=plot)

    ## fit tps on test data:
    x_pred = f_tps(x_test)
    
    ## plot:
    if plot:
        for i in xrange(3):
            plt.subplot(3,1,i+1)
            plt.plot(x_test[:,i],label='hydra')
            plt.plot(x_gt_test[:,i],label='cam')
            plt.plot(x_pred[:,i],label='pred')
            plt.legend()
        plt.show()


import argparse        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', help="Plot the data", action="store_true", default=False)
    parser.add_argument('--rviz', help="Publish the data on topics for visualization on RVIZ", action="store_true", default=True)
    parser.add_argument('-freq', help="frequency in filter", action='store', dest='freq', default=30., type=float)
    parser.add_argument('-dname', help="name of demonstration file", action='store', dest='demo_fname', default='demo100', type=str)
    parser.add_argument('-clib', help="name of calibration file", action='store', dest='calib_fname', default = 'cc_two_camera_calib', type=str)
    vals = parser.parse_args()
    
    #rospy.init_node('viz_demos',anonymous=True)    

    freq        = vals.freq
    demo_fname  = vals.demo_fname
    calib_fname = vals.calib_fname
    
    demo_dir  = hd_path + '/hd_data/demos/' + demo_fname;
    data_file = osp.join(demo_dir, 'demo.data')
    
    fit_and_plot_tps(True)
    #fit_gpr(data_file, calib_fname, freq, use_spline=False, customized_shift=None, single_camera=True, plot_commands='1h')
