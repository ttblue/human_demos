"""
Runs the kalman filter on observations from the hydra.
"""
from __future__ import division
import cPickle
import numpy as np
from hd_track.kalman import kalman
from hd_track.kalman import smoother
from hd_utils import transformations as tfms
import argparse
import matplotlib.pylab as plt
import os.path as osp
from kalman_tuning import state_from_tfms, closer_angle, state_from_tfms_no_velocity
import scipy.linalg as scl
from l1 import l1
import cvxopt as cvx

hd_path = '/home/sibi/sandbox/human_demos'
#hd_path = '/home/ankush/sandbox444/human_demos'

def run_kalman_filter(T_hydra, T_ar, x_init, covar_init, ar_cov_scale, hydra_cov_scale, f=30.):
    """
Runs the kalman filter
"""
    dt = 1/f
    N = len(T_hydra)
    
    ## load the noise covariance matrices:
    covar_mats = cPickle.load(open(hd_path + '/hd_track/data/nodup-covars-1.cpickle'))
    motion_covar = covar_mats['process']
    hydra_covar = hydra_cov_scale*covar_mats['hydra']
    ar1_covar = ar_cov_scale*covar_mats['kinect']
    ar2_covar = ar_cov_scale*covar_mats['kinect']

    ## initialize the filter:
    KF = kalman()
    KF.init_filter(0, x_init, covar_init, motion_covar, hydra_covar, ar1_covar, ar2_covar)
    
    ts = dt * (np.arange(N)+1)
    filter_estimates = [] ## the kalman filter estimates
    filter_covariances = [] ## the kalman filter covariances
    ## run the filter:
    for i in xrange(len(ts)-1):
        KF.register_observation(ts[i], None, T_ar[i+1], T_hydra[i+1])
        filter_estimates.append(KF.x_filt)
        filter_covariances.append(KF.S_filt)
    
    A, R = KF.get_motion_mats(dt)
    


    return filter_estimates, filter_covariances, A, R


def plot_kalman(X_kf, X_ks, X_bh_hg, X_bg, X_ba_ag_t, valid_stamps):
    """
Plots the Kalman filter belief (X_kf), the observed states (X_bh),
the true states (from PR2, X_bg_gh).
"""
    
    assert len(X_kf) == len(X_bh_hg) == len(X_bg), "The number of state vectors are not equal. %d, %d, %d"%(len(X_kf), len(X_bh_hg), len(X_bg))
    
    to_plot=[0,1,2,6,7,8]
    axlabels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw', 'v_roll', 'v_pitch', 'v_yaw']
    for i in to_plot:
        plt.subplot(4,3,i+1)
        plt.plot(X_kf[i,:], label='filter')
        plt.plot(X_ks[i,:], label='smoother')
        plt.plot(X_bh_hg[i,:], label='hydra')
        plt.plot(X_bg[i,:], label='pr2')
        plt.plot(valid_stamps, X_ba_ag_t[i,:], '.', label='ar_marker')
        plt.ylabel(axlabels[i])
        plt.legend()
    

def load_data():

    ## load data from both cameras and hydra:
    dat = cPickle.load(open(hd_path + '/hd_data/demos/obs_data/demo1.data'))
    Ts_c1 = [tup[0] for tup in dat['camera1']]
    ts_c1 = [tup[1] for tup in dat['camera1']]
    print ts_c1[0]
    print len(ts_c1)
   

def run_kalman_and_plot(ar_cov_scale, hydra_cov_scale):
    """
    Runs the kalman filter and plots the results.
`    """
    dt = 1/30.
    Ts_bh, Ts_bg, T_gh, Ts_bh_hg, X_bg, X_bh_hg, Ts_ba_ag, X_ba_ag_t, ar_valid_stamps  = load_data()

    ## initialize the kalman belief:
    x_init = X_bg[:,0]
    I3 = np.eye(3)
    S_init = scl.block_diag(1e-6*I3, 1e-3*I3, 1e-4*I3, 1e-1*I3)

    ## run the kalman filter:
    Ts_obs = [np.eye(4) for _ in xrange(len(Ts_bh))]
    f_estimates, f_covariances, A, R = run_kalman_filter(Ts_bh_hg, Ts_ba_ag, x_init, S_init,ar_cov_scale, hydra_cov_scale, 1./dt)


    X_kf = np.array(f_estimates)
    X_kf = np.reshape(X_kf, (X_kf.shape[0], X_kf.shape[1])).T

    ## plot the results:
    #plot_kalman(X_kf[:,1:], X_bh_hg, X_bg, X_ba_ag_t, ar_valid_stamps)
    #plt.show()

    ## run the kalman smoother:
    f_covs = []
    for f_cov in f_covariances:
        f_covs.append(1e-3 * f_cov)
    s_estimates = smoother(A, R, f_estimates, f_covs)

    X_ks = np.array(s_estimates)
    X_ks = np.reshape(X_ks, (X_ks.shape[0], X_ks.shape[1])).T

    ## plot the results:
    plot_kalman(X_kf[:,1:], X_ks[:,1:], X_bh_hg, X_bg, X_ba_ag_t, ar_valid_stamps)
    plt.show()
