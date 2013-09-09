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
from kalman_tuning import state_from_tfms, closer_angle
import scipy.linalg as scl
from l1 import l1
import cvxopt as cvx 

data_dir = '/home/ankush/sandbox444/human_demos/hd_track/data/'


def run_kalman(T_obs, x_init, covar_init, f=30.):
    """
    Runs the kalman filter using just the observations from hydra.
    """
    dt = 1/f
    N = len(T_obs)
    
    ## load the noise covariance matrices:
    covar_mats = cPickle.load(open(osp.join(data_dir, 'covars-xyz-rpy.cpickle')))
    motion_covar = covar_mats['process']
    hydra_covar  = 1e5*covar_mats['hydra']

    ## initialize the filter:
    KF = kalman()  
    KF.init_filter(0, x_init, covar_init, motion_covar, hydra_covar)
    
    ts = dt * (np.arange(N)+1)
    estimates = []   ## the kalman filter estimates

    ## run the filter:
    for i in xrange(len(ts)-1):
        KF.observe_hydra(T_obs[i+1], ts[i])
        estimates.append(KF.x_filt)
    
    return estimates


def plot_kalman(X_kf, X_bh, X_bg_gh):
    """
    Plots the Kalman filter belief (X_kf), the observed states (X_bh),
    the true states (from PR2, X_bg_gh). 
    """
    
    assert len(X_kf) == len(X_bh) == len(X_bg_gh), "The number of state vectors are not equal. %d, %d, %d"%(len(X_kf), len(X_bh), len(X_bg_gh))
    
    axlabels = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'roll', 'pitch', 'yaw', 'v_roll', 'v_pitch', 'v_yaw']
    for i in range(12):
        plt.subplot(4,3,i+1)
        plt.plot(X_kf[i,:], label='kalman')            
        plt.plot(X_bh[i,:], label='hydra')
        plt.plot(X_bg_gh[i,:], label='pr2')
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
    
    """
    n = len(X)
    amat = np.empty((n-k+1, 2*k))
    for i in xrange(n-k+1):
        amat[i,:] = np.r_[X[i : i+k], np.square(X[i : i+k])] 
    return amat


def fit_auto(X,Y, k, do_l1=False):
    """
    Yi = [Xi-1  Xi-2 ... Xi-1-k]*[a1 a2 ... ak]T
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
            j = i+4 if i > 2 else i
            Ai, bi, W[i,:] = fit_auto(X_bh[j,:], X_bg_gh[j,:], k, do_l1)
            est = Ai.dot(W[i,:])
            print "  norm err : ", np.linalg.norm(bi - est)
    
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
    dat = cPickle.load(open(osp.join(data_dir, 'good_calib_hydra_pr2/pr2-hydra-combined-dat.cpickle')))
    Ts_bh = dat['Ts_bh']
    Ts_bg = dat['Ts_bg']
    T_gh = cPickle.load(open(osp.join(data_dir, 'good_calib_hydra_pr2/T_gh')))
        
    assert len(Ts_bg) == len(Ts_bh), "Number of hydra and pr2 transforms not equal."
    Ts_bg_gh = [t.dot(T_gh) for t in Ts_bg]   
    
    X_bh    = state_from_tfms(Ts_bh, dt).T
    X_bg_gh = state_from_tfms(Ts_bg_gh, dt).T


    X_bg_gh[6:9,:] = closer_angle(X_bg_gh[6:9,:], X_bh[6:9,:])

    return (Ts_bh, Ts_bg, T_gh, Ts_bg_gh, X_bh, X_bg_gh) 


def run_kf_and_plot():
    """
    Runs the kalman filter and plots the results.
    """
    dt = 1/30.
    Ts_bh, Ts_bg, T_gh, Ts_bg_gh, X_bh, X_bg_gh = load_data()
   
    ## initialize the kalman belief: 
    x_init = X_bg_gh[:,0]
    I3 = np.eye(3)
    S_init = scl.block_diag(1e-6*I3, 1e-3*I3, 1e-4*I3, 1e-1*I3)
    
    ## run the kalman filter:
    X_kf = run_kalman(Ts_bh, x_init, S_init, 1./dt)
    
    X_kf = np.array(X_kf)
    X_kf = np.reshape(X_kf, (X_kf.shape[0], X_kf.shape[1])).T
    
    ## plot the results:
    plot_kalman(X_kf[:,1:], X_bh, X_bg_gh)
    plt.show()
