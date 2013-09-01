from __future__ import division
import hd_utils.transformations as tfm
from hd_utils.colorize import colorize
import numpy as np
import scipy.linalg as scl


class kalman:
    """
    Simple Kalman filter to track 6DOF pose.
      - 12 state variables : x, vel_x, rot, vel_rot
      - Rotations are represented as Euler angles : proper care is taken to avoid singularities.
    """

    def __init__(self):

        # store all the estimates ('y's) till now.
        self.ys = []

        ## standard deviations of the state variables in control update:
        ## at least this much uncertainty is injected in each update:
        self.min_x_std  = 0.005 # m
        self.min_vx_std = 0.005 # m/s
        self.min_r_std  = 1     # deg
        self.min_vr_std = 1     # deg/s

        # time-dependent uncertainty : as time grows, so does the uncertainty:
        self.x_std_t  = 1  # m/s
        self.vx_std_t = 1  # m/s /s
        self.r_std_t  = 90 # deg/s
        self.vr_std_t = 90 # deg/s /s

      
        ## standard deviations of the measurements:
        self.ar_x_std     = 0.001 # m / sample
        self.ar_r_std     = 30    # deg / sample
        self.hydra_vx_std = 0.01 # m/s / sample
        self.hydra_r_std  = 0.1  # deg/ sample
        
        ## convert the above numbers to radians:
        self.min_r_std   *= (np.pi/180.)        
        self.min_vr_std  *= (np.pi/180.)
        self.r_std_t     *= (np.pi/180.)
        self.vr_std_t    *= (np.pi/180.)
        self.ar_r_std    *= (np.pi/180.)
        self.hydra_r_std *= (np.pi/180.)

        # update frequency : the kalman filter updates the estimate explicitly at this rate.
        # it also updates when a measurement is given to it.
        self.freq = 30.
    
        ## last observations : used to calculate observation velocities.
        self.hydra_prev = None
        self.ar_prev    = None

        ## the filter's current belief and its time:
        self.t_filt = None
        self.x_filt = None
        self.S_filt = None


    def get_motion_covar(self, t):
        """
        Returns the noise covariance for the motion model.
        Assumes a diagonal structure for now.
        """
        covar = np.eye(12)
        sq = np.square
        covar[0:3,0:3]   *= sq(max( self.x_std_t * t,  self.min_x_std ))
        covar[3:6,3:6]   *= sq(max( self.vx_std_t * t, self.min_vx_std))
        covar[6:9,6:9]   *= sq(max( self.r_std_t * t,  self.min_r_std ))
        covar[9:12,9:12] *= sq(max( self.vr_std_t * t, self.min_vr_std))
        return covar


    def get_motion_mat(self, t):
        """
        Return the matrix A in x_t+1 = Ax_t + n_t.
        t is the time-lapsed, i.e. the value of (t+1 - t).
        
        Return the matrix corresponding to a state of (x, vel_x, rot, vel_rot).
        """
        m = np.eye(6)
        m[0:3, 3:6] = t*np.eye(3)
        return scl.block_diag(m,m)


    def get_motion_mats(self, t):
        return (self.get_motion_mat(t), self.get_motion_covar(t))


    def get_hydra_mats(self, pos_vel=True):
        """
        Returns a tuple : 1. The observation matrix mapping the state to the observation.
                          2. The noise covariance matrix

        Hydras observe rotations and the translation velocities to high accuracy.

        Assumed that we observe the translation velocity and the rotation position from the hydras.
        Absolute position from the hydras is not that great.
        
        If pos_vel is True : then the translation velocity is also included in the observation.
                             otherwise only the rotational position is included.
        """
        if pos_vel: # observe translation velocity and rotation
            cmat = np.zeros(6,12)
            cmat[0:3, 3:6] = np.eye(3)
            cmat[3:6, 6:9] = np.eye(3)
            
            vmat = np.eye(6)
            vmat[0:3,0:3] *= (self.hydra_vx_std*self.hydra_vx_std)
            vmat[3:6,3,6] *= (self.hydra_r_std*self.hydra_r_std) 
            return (cmat, vmat)
        
        else: # can observe only the rotation
            cmat = np.zeros(3,12)
            cmat[0:3, 6:9] = np.eye(3)
            vmat = (self.hydra_r_std*self.hydra_r_std)*np.eye(3) 
            return (cmat, vmat)
        

    def get_ar_mats(self):
        """
        Returns a tuple : observation matrix and the corresponding noise covariance matrix
                          for AR marker observations.
        AR markers observe the translation to very high accuracy, but rotations are very noisy. 
        """
        cmat = np.zeros(6,12)
        cmat[0:3,0:3] = np.eye(3)
        cmat[3:6,6:9] = np.eye(3)
        
        vmat = np.eye(6)
        vmat[0:3,0:3] *= (self.ar_x_std*self.ar_x_std)        
        vmat[3:6,3:6] *= (self.ar_r_std*self.ar_r_std)
        
        return (cmat, vmat)


    def control_update(self, x_p, S_p, t=None):
        """
        Runs the motion model forward by time t, assuming previous mean is x_p
        and previous covariance is S_p.
        Returns the next mean and covariance (x_n, S_n).
        """
        if t == None:
            t = 1./self.freq
        
        A, R = self.get_motion_mats(t)
        x_n = A.dot(x_p)
        S_n = A.dot(S_p).dot(A.T) + R
        return (x_n, S_n)

    
    def measurement_update(self, z_obs, C_obs, Q_obs, x_b, S_b):
        """
        z_obs        : Measurement vector
        C_obs, Q_obs : Measurement matrix and measurement noise covariance.
        x_b, S_b     : control-prediction of mean and covariance.
        
        returns the updated mean x_n and the covariance S_n.
        """
        L = np.linalg.inv(C_obs.dot(S_b).dot(C_obs.T) + Q_obs)
        K = S_b.dot(C_obs.T).dot(L)
        
        x_n = x_b + K.dot(z_obs - C_obs.dot(x_b))
        S_n = S_b - K.dot(C_obs).dot(S_b)
        
        return (x_n, S_n)
    

    def observe_ar(self, tfm, t):
         
    
    
    