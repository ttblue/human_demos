from __future__ import division

import numpy as np
import os, os.path as osp
import cPickle as cp
import pickle
import matplotlib.pylab as plt

from hd_utils.colorize import colorize
from hd_utils.conversions import *
from hd_utils.utils import *
from hd_utils.defaults import tfm_link_rof


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

def fit_tps(x_gt, x, plot=True, save_fname=None):
    """
    Fits a thin-plate spline model to x (source points) and x_gt (ground-truth target points).
    This transform can be used to correct the state-dependent hydra errors.
    
    X, X_GT are of the shape : Nx3
    
    """
    bend_coef = 0.0  ## increase this to make the tps-interpolation more smooth
    f_tps = registration.fit_ThinPlateSpline(x, x_gt, bend_coef = bend_coef, rot_coef = 0.0)

    if plot:
        plot_reqs = plot_warping(f_tps.transform_points, x, x_gt, fine=False, draw_plinks=True)
        plotter   = PlotterInit()
        for req in plot_reqs:
            plotter.request(req)

    if save_fname != None:
        save_tps(f_tps, save_fname)

    return f_tps


def save_tps(f_tps, fname):
    f_dat = {'x_na'   : f_tps.x_na,
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
    return f_tps


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


def correct_hydra(Ts_hydra, T_tt2hy, T_cam2hbase, f_tps):
    """
    Warps the xyz from hydra according to f_tps.
    Note the rpy are left unchanged.
    ================================

    Ts_hydra  : Tool-tip transform from hydra's estimate in cam1 frame
    T_tt2hy   : Transform from tool-tip to hydra-sensor on the gripper
    T_cam2hbase : Transform from camera1 to hydra-base
    """
    T_hbase2cam = np.linalg.inv(T_cam2hbase)
    T_hy2tt     = np.linalg.inv(T_tt2hy)

    Ts_HB = put_cam_in_hbase(Ts_hydra, T_tt2hy, T_cam2hbase)
    N     = len(Ts_HB)
    Xs_HB = np.empty((N,3))
    for i in xrange(N):
        Xs_HB[i,:] = Ts_HB[i][0:3,3]
    Xs_aligned     = f_tps.transform_points(Xs_HB)

    Ts_aligned = []    
    for i in xrange(N):
        t_aligned = Ts_HB[i]
        t_aligned[0:3,3] = Xs_aligned[i,:]
        t_aligned_cam2tt = T_cam2hbase.dot(t_aligned).dot(T_hy2tt)
        Ts_aligned.append(t_aligned_cam2tt)

    return Ts_aligned


def fit_tps_on_tf_data(Ts_hy, Ts_cam, T_tt2hy, T_cam2hbase, plot=False):
    """
    Given two lists of CORRESPONDING transforms as saved in extract_data.py
    (i.e., these are the estimates of the tool-tip in camera1 frame,
           using:
           Hydra for Ts_hy
           Camera for Ts_cam
    """
    Ts_hy_hbase  = put_cam_in_hbase(Ts_hy, T_tt2hy, T_cam2hbase)
    Ts_cam_hbase = put_cam_in_hbase(Ts_cam, T_tt2hy, T_cam2hbase)
    n_matching   = len(Ts_hy_hbase)
    x_hy, x_cam  = np.empty((n_matching,3)), np.empty((n_matching,3))
    for i in xrange(n_matching):
        x_hy[i,:]  = Ts_hy_hbase[i][0:3,3]
        x_cam[i,:] = Ts_cam_hbase[i][0:3,3]
    
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
