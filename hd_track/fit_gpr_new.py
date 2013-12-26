from __future__ import division

"""
Code to do GPR state-dependent error fitting.
Uses kalman filtering code to align the hydra and the camera data-streams.
"""

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


from hd_track.kalman import kalman, closer_angle
from hd_track.kalman import smoother


from hd_track.kalman_tuning import state_from_tfms_no_velocity
from hd_track.streamer import streamize
from hd_track.stream_pc import streamize_pc, streamize_rgbd_pc
from hd_visualization.ros_vis import draw_trajectory 

import hd_utils.transformations as tfms
from hd_utils.defaults import calib_files_dir

## for GPR:
#import error_characterization as ec
from GPR import gpr
import pickle
import openravepy as rave
from Tools.general import feval


from tps_correct import fit_tps

hd_path = os.getenv('HD_DIR')


class hd_gpr:
    """
    For Gaussian Process Regression (GPR) state-dependent error correction
    for hydra's pose estimates.
    
    Basically a function approximator.
    
    Training:
    =========
        1. Need to train two things : log-hyper parameters and "alphas"
        2. Does cross validation to prevent over-fitting.
    
    Testing/ Correction:
    ====================
        1. Given an input transform(s), it spits out its prediction and standard deviation. 
    
    Might need to run a low-pass filter/ use the covariances given by this GPR predictions
    in the Kalman Filter for better performance.
    
    Note: It makes sense to correct only the xyz of the hydras.
                                            =====
          The pose estimates from the hydra are better than those from the camera 
          hence, we need only correct for hydra's position estimates.
          
          The state-vector to used for training should be (xyz and axis-angles).
    """

    def __init__(self):
        
        ## the following training data is instantiated once the GPR model has been trained.
        self.trained    = False
        self.train_data = {}
        self.train_data['Xs_gt']        = None  # ground-truth positions
        self.train_data['Xs_noisy']     = None  # noisy positions
        self.train_data['state_gt']     = None  # ground-truth state-vectors <<-- don't really need these.
        self.train_data['state_noisy']  = None  # noisy-state-vectors
        self.train_data['loghyper']     = None
        self.train_data['alphas']       = None
        self.train_data['T_sys']        = None  # the systematic pose-offset


    def __precompute_alpha(self, logtheta, covfunc, X, y):
        # compute training set covariance matrix (K)
        K = feval(covfunc, logtheta, X)                # training covariances
        L = np.linalg.cholesky(K)                      # cholesky factorization of cov (lower triangular matrix)
        alpha = gpr.solve_chol(L.transpose(),y)        # compute inv(K)*y
        return alpha

    def __gpr_precompute(self, x_gt, x_noisy, state_train):
        """
        Compute the alphas and hyper-params on the training-data.
    
        x_gt   : ground-truth positions (nx3) matrix.
        x_noisy: noisy positions (nx3) matrix.
        state_train : the states corresponding to x_noisy (nx6) matrix (the state is xyzrpy). 
        """
        N = x_gt.shape[0]
        assert x_noisy.shape[0]==N==state_train.shape[0]
    
        x_errors  = x_gt - x_noisy

        alphas      = []
        hparams     = np.empty((3,3))
        covfunc     = ['kernels.covSum', ['kernels.covSEiso','kernels.covNoise']]
        n_task_vars = 3

        for i_task_var in range(n_task_vars):
            print 'GP training variable : ', (i_task_var+1), '/', n_task_vars
            logtheta = np.array([-1,-1,-1])
            X        = state_train
            y        = x_errors[:, i_task_var]
            y        = np.reshape(y, (N,1))
            hparams[i_task_var,:] = gpr.gp_train(logtheta, covfunc, X, y)
            alphas.append(self.__precompute_alpha(hparams[i_task_var,:], covfunc, X, y))

        return [hparams, alphas]


    def sys_correct(self, X_gt, X):
        """
        X_gt : 3xn matrix of ground-truth positions.
        X    : 3xn matrix of positions to be aligned.
    
        Returns a transform T such that X_gt = T*X
        """
        X_gt_mean = np.mean(X_gt, axis=1)
        X_mean    = np.mean(X, axis=1)
    
        X_cent    = X - X_mean[:,None]
        X_gt_cent = X_gt - X_gt_mean[:,None]
    
        S         = X_cent.dot(X_gt_cent.T)
        U,_,V     = np.linalg.svd(S)
        I_align   = np.eye(3)
        I_align[2,2] = np.linalg.det(V.dot(U.T))

        R_opt     = V.dot(I_align.dot(U.T))
        t_opt     = X_gt_mean - R_opt.dot(X_mean)
    
        return (R_opt, t_opt)


    def train(self, Ts_gt, Ts_noisy):
        """
        Train the GPR.
        Ts_gt    : A list of ground-truth poses.
        Ts_noisy : A list of noisy poses. 
        """
        
        ## extract data in the required format:
        N = len(Ts_gt)
        assert N==len(Ts_noisy), "GPR training : lengths are not the same."
        Xs_gt       = state_from_tfms_no_velocity(Ts_gt)
        Xs_noisy    = state_from_tfms_no_velocity(Ts_noisy)
        
        xs_noisy    = Xs_noisy[:,0:3]
        xs_gt       = Xs_gt[:,0:3] 
       
        state_noisy = np.c_[Xs_noisy[:,0:3], Xs_noisy[:,6:9]]

        self.train_data['Xs_gt']        = xs_gt
        self.train_data['Xs_noisy']     = xs_noisy
        self.train_data['state_noisy']  = state_noisy

        ## fit systematic transform (using SVD):
        R_sys, t_sys = self.sys_correct(xs_gt.T, xs_noisy.T)
        T_sys = np.eye(4)
        T_sys[0:3,0:3] = R_sys
        T_sys[0:3,3]   = t_sys
        self.train_data['T_sys'] = T_sys
        print "Systematic transform : ", T_sys

        xs_sys_corrected = R_sys.dot(xs_noisy.T) + t_sys[:,None]
        xs_sys_corrected = xs_sys_corrected.T
        
        ## fit GPR model:
        hparams, alphas = self.__gpr_precompute(xs_gt, xs_sys_corrected, state_noisy)
        self.train_data['loghyper'] = hparams
        self.train_data['alphas']   = alphas    
        self.trained                = True
        

        # same as gp_pred except that uses the precomputed alpha (what is returned from gp_pred_precompute_alpha())
    def __gpr_pred_fast(self,  X, Xstar, covfunc, alpha, logtheta):
        [Kss, Kstar] = feval(covfunc, logtheta, X, Xstar)
        return np.dot(Kstar.transpose(),alpha)


    def __gpr_correct(self, x_sys, state_in):
        """
        Applied the GPR model on the system corrected x_sys (nx3) corresponding
        to the states state_in (nx6).
        
        Returns the corrected xyz (nx3) matrix. 
        """
        N, n_task_vars  = x_sys.shape[0], x_sys.shape[1] 
        MU = np.empty(x_sys.shape)
        covfunc = ['kernels.covSum', ['kernels.covSEiso','kernels.covNoise']]
        X     = self.train_data['state_noisy']
        Xstar = state_in

        for i_task_var in range(n_task_vars):

            loghyper_var_i = self.train_data['loghyper'][i_task_var,:] 
            alpha = self.train_data['alphas'][i_task_var]

            ## PREDICTION 
            print 'GP predicting variable :', (i_task_var+1), "/",n_task_vars
            MU[:,i_task_var]  = self.__gpr_pred_fast(X, Xstar, covfunc, alpha, loghyper_var_i)[:,0]

        return x_sys + MU

    
    def correct_poses(self, Ts_in):
        """
        Returns a list of corrected transforms.
        Ts_in is a list of transforms to correct.
        
        Note : Only the xyz are corrected.
        """
        if not self.trained:
            print "GPR model not trained. Cannot predict before training. Returning..."
            return

        state_in = state_from_tfms_no_velocity(Ts_in)
        state_in = np.c_[state_in[:,0:3], state_in[:,6:9]]  ## get rid of position and rotation velocities.
        
        # systematic correction:
        T_sys         = self.train_data['T_sys']
        R_sys, t_sys  = T_sys[0:3,0:3], T_sys[0:3,3]
        
        xs_in           = [tfm[0:3,3] for tfm in Ts_in]
        x_sys_corrected = [R_sys.dot(x) + t_sys for x in xs_in]
        x_sys_corrected = np.array(x_sys_corrected)

        # GP correction:
        x_gp_corrected = self.__gpr_correct(x_sys_corrected, state_in)
        Ts_corrected    = []
        for i,tfm in enumerate(Ts_in):
            tfm[0:3,3] = x_gp_corrected[i,:]
            Ts_corrected.append(tfm)
        return Ts_corrected

    def save_gpr_model(self, fname):
        """
        Save the GPR model to a file.
        """
        pickle.dump(self.train_data, open(fname, 'wb'))

    def load_gpr_model(self, fname):
        """
        Load the trained model from a file. 
        If loaded from a file, then there is no need to re-train the model.
        """
        self.train_data = pickle.load(open(fname, 'rb'))
        self.trained = True


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
    hydra_covar  = 1e-4*np.eye(6) # for rpy 1e-4 
    hydra_covar[0:3,0:3] = 1e-2*np.eye(3) # for xyz 1e-2
    
    hydra_vcovar = 1e-3*np.eye(6) # for xyz-v 1e-5

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
    """
    return start-end time, number of time samples, and streams for each sensor
    """
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
    plt.legend()

def correlation_shift(xa,xb):
    shifts = []
    for idx in [0, 1, 2]:
        shifts.append(np.argmax(np.correlate(xa[idx,:], xb[idx,:], 'full'))-(xb.shape[1]-1))
    print shifts
    return  int(np.max(shifts))

  
def get_synced_data(data_file, calib_file, lr, freq, use_spline, customized_shift, single_camera, plot=False):
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


def rpy2axang(rpy):
    """
    Converts a matrix of rpy (nx3) into a matrix of 
    axis-angles (nx3). 
    """
    n = rpy.shape[0]
    assert rpy.shape[1]==3, "unknown shape."
    ax_ang = np.empty((n,3))
    for i in xrange(n):
        th = rpy[i,:]
        ax_ang[i,:] = rave.axisAngleFromRotationMatrix(tfms.euler_matrix(th[0], th[1], th[2]))
    return ax_ang


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


def fit_gpr(data_file, calib_file, freq, use_spline=False, customized_shift=None, single_camera=False, plot_commands='s12fh'):
    #dat = cp.load(open(data_file))
    #if dat.has_key('r'):
    if True:
        """
        T_hy, T_cam_gt = get_synced_data(data_file, calib_file, 'r', freq, use_spline, customized_shift, single_camera)
        f = open('grp.dat', 'w')
        pickle.dump([T_cam_gt, T_hy], f)
        f.close()
        """
        T_cam_gt, T_hy = pickle.load(open('grp.dat', 'r'))        

        BLOCK_SIZE  = 10
        N           = len(T_hy)
        train, tst  = gen_indices(N, BLOCK_SIZE)

        T_hy_train     = [T_hy[i] for i in train]
        T_hy_test      = [T_hy[i] for i in tst]
        T_cam_gt_train = [T_cam_gt[i] for i in train]
        T_cam_gt_test  = [T_cam_gt[i] for i in tst]      

        """
        #print "tuning hyper-params.."
        #hi_params = ec.train_hyperparams(T_cam_gt_train, T_hy_train, state_train)
        #cp.dump(hi_params, open('hyper-params.cpkl', 'wb')) ## save the hyper-parameters to a file.

        hi_params = cp.load(open('hyper-params.cpkl'))
        print hi_params

        print "GRP correcting..."
        T_test_ests  = ec.gp_correct_poses(T_cam_gt_train, T_hy_train, state_train, T_hy_test, state_test, hi_params)
        print "  .. done"
        
        X_est = state_from_tfms_no_velocity(T_test_ests).T
        X_est = np.c_[X_est[:,0], X_est]
        ###########################################
        """
        gpr_model = hd_gpr()
        #gpr_model.train(T_cam_gt_train[0:10], T_hy_train[0:10])
        gpr_model.train(T_cam_gt, T_hy)
        gpr_model.save_gpr_model("gpr_model.pkl")
        #gpr_model.load_gpr_model("gpr_model.pkl")
        T_corr_test = gpr_model.correct_poses(T_hy_test)
        ## plot the corrected poses:
        to_plot  = [0,1,2,6,7,8] # for xyz rpy
        axlabels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        
        X_hy_test     = state_from_tfms_no_velocity(T_hy_test).T
        X_cam_gt_test = state_from_tfms_no_velocity(T_cam_gt_test).T
        X_est         = state_from_tfms_no_velocity(T_corr_test).T
        
        for count,i in enumerate(to_plot):
            plt.subplot(2,3,count+1)
            plt.plot(X_hy_test[i,:], label='hydra')
            plt.plot(X_cam_gt_test[i,:], label='cam_gt')
            plt.plot(X_est[i,:], label='gpr_est')
            plt.ylabel(axlabels[count])
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
