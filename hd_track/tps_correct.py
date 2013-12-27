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
    """
    bend_coef = 0.00  ## increase this to make the tps-interpolation more smooth
    f_tps = registration.fit_ThinPlateSpline(x, x_gt, bend_coef = bend_coef, rot_coef = 0.001)

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
