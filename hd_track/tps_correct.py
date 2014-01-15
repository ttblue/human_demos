from __future__ import division

import numpy as np
import os, os.path as osp
import cPickle as cp
import pickle
import matplotlib.pylab as plt

from hd_utils.colorize import colorize
from hd_utils.conversions import *
from hd_utils.utils import *
from hd_utils.defaults import tfm_link_rof, hd_path, calib_files_dir, demo_files_dir, demo_names
from   hd_utils.tps_utils import *
from   hd_utils.mayavi_plotter import *

## for tps-correct:
from hd_rapprentice import registration



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
    
    X, X_GT are of the shape : Nx3
    
    """
    bend_coeff = 0.1  ## increase this to make the tps-interpolation more smooth, decrease to fit better.
    f_tps = registration.fit_ThinPlateSpline(x, x_gt, bend_coef = bend_coeff, rot_coef = 0.01*bend_coeff)

    if plot:
        plot_reqs = plot_warping(f_tps.transform_points, x, x_gt, fine=False, draw_plinks=True)
        plotter   = PlotterInit()
        for req in plot_reqs:
            plotter.request(req)

    return f_tps


def save_tps(f_tps, T_cam2hbase, fname):
    """
    Saves the tps-model to a file.
    T_cam2hbase is the calibration transform from the
    camera1 (main camera) to the hydra-base.
    """
    f_dat = {'T_cam2hbase_train':T_cam2hbase,
             'x_na'   : f_tps.x_na,
             'lin_ag' : f_tps.lin_ag,
             'trans_g': f_tps.trans_g,
             'w_ng'   : f_tps.w_ng}
    print "Writing tps model to %s"%fname
    cp.dump(f_dat, open(fname, 'wb'))


def load_tps(fname):
    with open(fname,'rb') as f:
        f_dat = cp.load(f)
    f_tps = registration.ThinPlateSpline()
    f_tps.x_na    = f_dat['x_na']
    f_tps.lin_ag  = f_dat['lin_ag']
    f_tps.trans_g = f_dat['trans_g']
    f_tps.w_ng    = f_dat['w_ng']
    return f_tps, f_dat['T_cam2hbase_train']


def put_cam_in_hbase(Ts_cam, T_tt2hy, T_cam2hbase):
    """
    Transforms the tfs in Ts_cam so that they are in hydra-base frame
    and give the hydra-sensor's pose estimate.
    """
    T_hbase2cam = np.linalg.inv(T_cam2hbase)
    Ts_new = []
    for tf in Ts_cam:
        Ts_new.append(T_hbase2cam.dot(tf).dot(T_tt2hy))
    return Ts_new

def put_hbase_in_cam(Ts_hbase, T_tt2hy, T_cam2hbase):
    """
    Transforms the tfs in Ts_hbase so that they are in camera frame.
    """
    T_hy2tt = np.linalg.inv(T_tt2hy)
    Ts_new = []
    for tf in Ts_hbase:
        Ts_new.append(T_cam2hbase.dot(tf).dot(T_hy2tt))
    return Ts_new


def correct_hydra(Ts_hydra, T_cam2hbase, f_tps, T_cam2hbase_train):
    """
    Warps the xyz from hydra according to f_tps.
    Note the rpy are left unchanged.
    ================================

    Ts_hydra  : Tool-tip transform from hydra's estimate in cam1 frame
    T_cam2hbase : Transform from camera1 to hydra-base
    f_tps     : The TPS warping function
    T_cam2hbase_train : The calibration transform from camera1 
                        to hydra-base when f_tps was trained
    """
    N = len(Ts_hydra)
    Xs = np.empty((N,3))
    Ts_aligned = []

    do_transform = not np.allclose(T_cam2hbase, T_cam2hbase_train)
    if do_transform:
        redprint("\t TPS-correction : The camera hydra-base calibration has changed. Transforming.. ")
        T_tf = T_cam2hbase_train.dot(np.linalg.inv(T_cam2hbase))
        Ti_tf = np.linalg.inv(T_tf)
        R_tf,t_tf    = T_tf[0:3,0:3], T_tf[0:3,3]
        Ri_tf, ti_tf = Ti_tf[0:3,0:3], Ti_tf[0:3,3] 

        for i in xrange(N):
            Xs[i,:] = R_tf.dot(Ts_hydra[i][0:3,3]) + t_tf

        ## do a tps-warp:    
        Xs_aligned = f_tps.transform_points(Xs)

        for i in xrange(N):
            tfm = Ts_hydra[i].copy()
            tfm[0:3,3] = Ri_tf.dot(tfm[0:3,3]) + ti_tf
            Ts_aligned.append(tfm)
    else:
        for i in xrange(N):
            Xs[i,:] = Ts_hydra[i][0:3,3]

        ## do a tps-warp:
        Xs_aligned = f_tps.transform_points(Xs)

        for i in xrange(N):
            Ts_aligned.append(Ts_hydra[i].copy())
            Ts_aligned[i][0:3,3] = Xs_aligned[i,:]

    return Ts_aligned


def fit_tps_on_tf_data(Ts_hy, Ts_cam, plot=False):
    """
    Given two lists of CORRESPONDING transforms as saved in extract_data.py
    (i.e., these are the estimates of the tool-tip in camera1 frame,
           using:
           Hydra for Ts_hy
           Camera for Ts_cam)
    """
    n_matching   = len(Ts_hy)
    x_hy, x_cam  = np.empty((n_matching,3)), np.empty((n_matching,3))
    for i in xrange(n_matching):
        x_hy[i,:]  = Ts_hy[i][0:3,3]
        x_cam[i,:] = Ts_cam[i][0:3,3]

    return fit_tps(x_cam, x_hy, plot)
    

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
    parser.add_argument('-demo_name', help="name of demonstration", action='store', type=str)
    parser.add_argument('-demo_type', help="type of demonstration", action='store', type=str)
    vals = parser.parse_args()
    
    #rospy.init_node('viz_demos',anonymous=True)    

    freq        = vals.freq
    demo_dir  = osp.join(demo_files_dir, vals.demo_type, vals.demo_name)
    data_file = osp.join(demo_dir, demo_names.data_name)
    
    fit_and_plot_tps(True)
    #fit_gpr(demo_dir, freq, use_spline=False, customized_shift=None, single_camera=True, plot_commands='1h')
