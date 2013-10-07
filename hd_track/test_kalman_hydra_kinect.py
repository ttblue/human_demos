"""
Runs the kalman filter on observations from the hydra.
"""
from __future__ import division
import cPickle
import numpy as np
from hd_track.kalman import kalman
from hd_utils import transformations as tfms
import argparse
import matplotlib.pylab as plt
import os.path as osp
from kalman_tuning import state_from_tfms, closer_angle, state_from_tfms_no_velocity
import scipy.linalg as scl
from l1 import l1
import cvxopt as cvx

hd_path = '/home/henrylu/henry_sandbox/human_demos'


def run_kalman(T_hydra, T_ar, x_init, covar_init, f=30.):
    """
Runs the kalman filter using just the observations from hydra.
"""
    dt = 1/f
    N = len(T_hydra)
    
    ## load the noise covariance matrices:
    covar_mats = cPickle.load(open(hd_path + '/hd_track/data/timed-covars.cpickle'))
    motion_covar = covar_mats['process']
    hydra_covar = 1e5*covar_mats['hydra']
    ar_covar = 1e5*covar_mats['kinect']

    ## initialize the filter:
    KF = kalman()
    KF.init_filter(0, x_init, covar_init, motion_covar, hydra_covar, ar_covar)
    
    ts = dt * (np.arange(N)+1)
    estimates = [] ## the kalman filter estimates

    ## run the filter:
    for i in xrange(len(ts)-1):
        KF.register_observation(ts[i], T_ar[i+1], T_hydra[i+1])
        estimates.append(KF.x_filt)
    
    return estimates


def plot_kalman(X_kf, X_bh_hg, X_bg, X_ba_ag_t, valid_stamps):
    """
Plots the Kalman filter belief (X_kf), the observed states (X_bh),
the true states (from PR2, X_bg_gh).
"""
    
    assert len(X_kf) == len(X_bh_hg) == len(X_bg), "The number of state vectors are not equal. %d, %d, %d"%(len(X_kf), len(X_bh_hg), len(X_bg))
    
    to_plot=[0,1,2,6,7,8]
    axlabels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw', 'v_roll', 'v_pitch', 'v_yaw']
    for i in to_plot:
        plt.subplot(4,3,i+1)
        plt.plot(X_kf[i,:], label='kalman')
        plt.plot(X_bh_hg[i,:], label='hydra')
        plt.plot(X_bg[i,:], label='pr2')
        plt.plot(valid_stamps, X_ba_ag_t[i,:], '.', label='ar_marker')
        plt.ylabel(axlabels[i])
        plt.legend()
    
    
def get_auto_mat(X, k):
    """
Returns a matrix of (n-k+1) x k dimensions,
which is the auto-correlation matrix.
X is an n-dimensional vector.
"""
    n = len(X)
    amat = np.empty((n-k+1, k))
    for i in xrange(n-k+1):
        amat[i,:] = X[i : i+k]
    return amat

def get_auto_mat2(X, k):
    """
Returns a matrix of (n-k+1) x 2k dimensions,
which is the auto-correlation matrix.
X is an n-dimensional vector.
squared terms are also included.
TODO : add cross-terms and see if this does any better?
"""
    n = len(X)
    amat = np.empty((n-k+1, 2*k))
    for i in xrange(n-k+1):
        amat[i,:] = np.r_[X[i : i+k], np.square(X[i : i+k])]
    return amat


def fit_auto(X,Y, k, do_l1=False):
    """
Yi = [Xi-1 Xi-2 ... Xi-1-k]*[a1 a2 ... ak]T
Solves for a_k's : auto-regression.
"""
    assert k < len(X), "order more than the length of the vector."
    assert X.ndim==1 and Y.ndim==1, "Vectors are not one-dimensional."
    assert len(X)==len(Y), "Vectors are not of the same size."
    
    A = get_auto_mat2(X, k)
    A = np.c_[A, np.ones(A.shape[0])]
    b = Y[k-1:]
    
    if do_l1:
        sol = np.array(l1(cvx.matrix(A), cvx.matrix(b)))
        sol = np.reshape(sol, np.prod(sol.shape))
    else:
        sol = np.linalg.lstsq(A, b)[0]
 
    return [A, b, sol]
    

def fit_calib_auto(X_bh, X_bg_gh, do_l1=False):
    """
Does auto-regression on the 6-DOF variables : x,y,z,r,p,y.
"""
    assert X_bh.shape[0]==X_bg_gh.shape[0]==12, "calib data has unknown shape."
    
    #X_bh = X_bh[:,500:1000]
    #X_bg_gh = X_bg_gh[:,500:1000]
    
    axlabels = ['x','y','z','roll','pitch','yaw']
    for k in xrange(1,100, 10):
        print "order : k=", k
        plt.clf()
        W = np.empty((6, 2*k+1))
        for i in xrange(6):
            j = i+3 if i > 2 else i
            Ai, bi, W[i,:] = fit_auto(X_bh[j,:], X_bg_gh[j,:], k, do_l1)
            est = Ai.dot(W[i,:])
            print " norm err : ", np.linalg.norm(bi - est)
    
            plt.subplot(3,2,i+1)
            plt.plot(bi, label='pr2')
            plt.plot(X_bh[j,k-1:], label='hydra')
            plt.plot(est, label='estimate')
            plt.ylabel(axlabels[i])
            plt.legend()
        plt.show()


def show_regress(do_l1=False):
    Ts_bh, Ts_bg, T_gh, Ts_bg_gh, X_bh, X_bg_gh = load_data()
    fit_calib_auto(X_bh, X_bg_gh, do_l1)



def load_data():

    dt = 1./30.

    ## load pr2-hydra calib data:
    dat = cPickle.load(open(hd_path + '/hd_track/data/timed-transforms-1.cpickle'))
    Ts_bh = dat['Ts_bh']
    Ts_bg = dat['Ts_bg']
    Ts_ba = dat['Ts_ba']
    T_gh = dat['T_gh']
    T_ga = dat['T_ga']

    assert len(Ts_bg) == len(Ts_bh), "Number of hydra and pr2 transforms not equal."
    Ts_bh_hg = [t.dot(np.linalg.inv(T_gh)) for t in Ts_bh]
    Ts_ba_ag = []
    ar_valid_stamps = []
    Ts_ba_ag_t = []
    for i in xrange(len(Ts_ba)):
        t = Ts_ba[i]
        if t == None:
            Ts_ba_ag.append(None)
        else:
            Ts_ba_ag.append(t.dot(np.linalg.inv(T_ga)))
            Ts_ba_ag_t.append(t.dot(np.linalg.inv(T_ga)))
            ar_valid_stamps.append(i)
    
    X_bg = state_from_tfms(Ts_bg, dt).T
    X_bh_hg = state_from_tfms(Ts_bh_hg, dt).T
    X_ba_ag_t = state_from_tfms_no_velocity(Ts_ba_ag_t).T
    ar_valid_stamps = ar_valid_stamps[1:]

    print X_bg.shape
    print X_ba_ag_t.shape
    print len(ar_valid_stamps)
    X_bh_hg[6:9,:] = closer_angle(X_bh_hg[6:9,:], X_bg[6:9,:])

    return (Ts_bh, Ts_bg, T_gh, Ts_bh_hg, X_bg, X_bh_hg, Ts_ba_ag, X_ba_ag_t, ar_valid_stamps)
    


def run_kf_and_plot():
    """
Runs the kalman filter and plots the results.
"""
    dt = 1/30.
    Ts_bh, Ts_bg, T_gh, Ts_bh_hg, X_bg, X_bh_hg, Ts_ba_ag, X_ba_ag_t, ar_valid_stamps  = load_data()
   
    ## initialize the kalman belief:
    x_init = X_bg[:,0]
    I3 = np.eye(3)
    S_init = scl.block_diag(1e-6*I3, 1e-3*I3, 1e-4*I3, 1e-1*I3)
    
    ## run the kalman filter:
    Ts_obs = [np.eye(4) for _ in xrange(len(Ts_bh))]
    X_kf = run_kalman(Ts_bh_hg, Ts_ba_ag, x_init, S_init, 1./dt)
    
    X_kf = np.array(X_kf)
    X_kf = np.reshape(X_kf, (X_kf.shape[0], X_kf.shape[1])).T
    
    ## plot the results:
    plot_kalman(X_kf[:,1:], X_bh_hg, X_bg, X_ba_ag_t, ar_valid_stamps)
    plt.show()
