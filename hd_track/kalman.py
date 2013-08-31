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
        self.x_std  = 1  # m/s
        self.vx_std = 1  # m/s /s
        self.r_std  = 90 # deg/s
        self.vr_std = 90 # deg/s /s
       
        ## standard deviations of the measurements:
        self.ar_x_std     = 0.001 # m / sample
        self.ar_r_std     = 30    # deg / sample
        self.hydra_vx_std = 0.01 # m/s / sample
        self.hydra_r_std  = 0.1  # deg/ sample
        
        ## convert the above numbers to radians:
        self.r_std       *= (np.pi/180.)
        self.vr_std      *= (np.pi/180.)
        self.ar_r_std    *= (np.pi/180.)
        self.hydra_r_std *= (np.pi/180.)

        # update frequency : the kalman filter updates the estimate explicitly at this rate.
        # it also updates when a measurement is given to it.
        self.freq = 30.


    def get_motion_covar(self, t):
        """
        Returns the noise covariance for the motion model.
        Assumes a diagonal structure for now.
        TODO ::: IMP : needs to have a constant offset too.
        """
        covar = np.eye(12)
        covar[0:3,0:3]   *= ((self.x_std  * self.x_std) * t)
        covar[3:6,3:6]   *= ((self.vx_std * self.vx_std) * t)
        covar[6:9,6:9]   *= ((self.r_std  * self.r_std) * t)
        covar[9:12,9:12] *= ((self.vr_std * self.vr_std) * t)
        return covar


    def get_motion_mat(self, t):
        """
        Return the matrix A in x_t+1 = Ax_t + n_t.
        t is the time-lapsed, i.e. the value of (t+1 - t).
        
        Return the matrix corresponding to a state of (x, vel_x, rot, vel_rot).
        """
        m = np.eye(6)
        m[0:3, 3:] = t*np.eye(3)
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


    def control_update(self, x_p, S_p, t):
        """
        Runs the motion model forward by time t, assuming previous mean is x_p
        and previous covariance is S_p.
        Returns the next mean and covariance (x_n, S_n).
        """
