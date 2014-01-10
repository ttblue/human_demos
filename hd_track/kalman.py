from __future__ import division
import numpy as np
import scipy.linalg as scl

import hd_utils.transformations as tfm
from hd_utils.colorize import colorize

def put_in_range(x):
    """
    Puts all the values in x in the range [-180, 180).
    """
    return ( (x+np.pi)%(2*np.pi) )- np.pi
            

def closer_angle(x, a):
    """
    returns the angle f(x) which is closer to the angle a in absolute value.
    """
    return a + put_in_range(x-a)



def canonicalize_obs(X_base, X_obs):
    """
    Returns the position and translation from T_obs (4x4 mat).
    Puts the rotation in a form which makes it closer to filter's
    current estimate in absolute terms.
    """
    rpy_base = X_base[6:9]
    rpy      = X_obs[6:9]
    rpy      = closer_angle(rpy, rpy_base)
    X_obs[6:9] = rpy
    return X_obs



def smoother(A, R, mu, sigma):
    """
    Kalman smoother implementation. 
    Implements the Rauch, Tung, and Striebel (RTS) smoother. 

    A    : Dynamics matrix, i.e. x_{t+1} = A*x_{t}
    R    : Dynamics noise covariance
    mu   : A list of time-series of the state means which are output of a kalman filter
    sigma: A list of state-covariances corresponding to the means above.

    Returns, (mu_smooth, sigma_smooth) : same size as mu, sigma.
    """
    assert len(mu) == len(sigma), "Kalman smoother : Number of means should be equal to the number of covariances."
    
    T = len(mu)
    ## prediction : x+t = Ax_t + r ~ N(0,R)
    mu_p    = [A.dot(x) for x in mu]
    sigma_p = [A.dot(S).dot(A.T) + R for S in sigma]

    conds = [np.linalg.cond(s) for s in sigma_p]
    print "condition min/max : ", np.min(conds), " , ", np.max(conds)

    mu_smooth    = [np.empty(mu[0].shape) for _ in xrange(len(mu))]
    sigma_smooth = [np.empty(sigma[0].shape) for _ in xrange(len(sigma))]

    # recursive smoother:
    #====================
    ## base case:  for last time-step
    mu_smooth[-1]    = mu[-1]
    sigma_smooth[-1] = sigma[-1]
    
    for t in xrange(T-2, -1, -1):
        L                   = sigma[t].dot(A.T).dot(np.linalg.inv(sigma_p[t]))
        mu_p_canon          = canonicalize_obs(mu_smooth[t+1], mu_p[t])
        mu_smooth[t]        = mu[t] + 0.9*(L.dot(mu_smooth[t+1] - mu_p_canon))
        mu_smooth[t][6:9,:] = put_in_range(mu_smooth[t][6:9,:])
        
        # added, though not necessary now
        sigma_smooth[t] = sigma[t] + L.dot(sigma_smooth[t+1] - sigma_p[t]).dot(L.T)
        
    for t in xrange(T):
        mu_smooth[t][6:9,:] = put_in_range(mu_smooth[t][6:9,:])

    return (mu_smooth, sigma_smooth)


class kalman:
    """
    Simple Kalman filter to track 6DOF pose.
      - 12 state variables : x, vel_x, rot, vel_rot
      - Rotations are represented as Euler angles : proper care is taken to avoid singularities.
    """

    def __init__(self):

        # store all the estimates ('y's) till now.
        self.ys = []
        
        # update frequency : the kalman filter updates the estimate explicitly at this rate.
        # it also updates when a measurement is given to it.
        self.freq = 30.

        ## the filter's current belief and its time:
        self.t_filt = None # time
        self.x_filt = None # mean
        self.S_filt = None # covariance
 
        self.motion_covar = None

        ## store the observation matrix for tf (hmat) based observations:
        ## NOTE: we observe just the xyz and the rpy from tf observations (no velocities).
        self.obs_mat = np.zeros((6,12))
        self.obs_mat[0:3, 0:3] = np.eye(3)
        self.obs_mat[3:6, 6:9] = np.eye(3)

   
    def canonicalize_obs(self, T_obs):
        """
        Returns the position and translation from T_obs (4x4 mat).
        Puts the rotation in a form which makes it closer to filter's
        current estimate in absolute terms.
        """
        pos = T_obs[0:3,3]
        rpy = np.array(tfm.euler_from_matrix(T_obs), ndmin=2).T
        rpy = closer_angle(rpy, self.x_filt[6:9])
 
        return (pos, rpy)


    def init_filter(self, t, x_init, S_init, motion_covar):
        """
        Give the initial estimate for the filter.
        t is the time of the estimate.
        x_init is a 12 dimensional state vector.
        S_init is the state covariance (12x12).
        
        motion_covar : noise-covariance for the motion model (12x12)
        cam_covar    : noise-covariance for the camera observations (6x6)
        hydra_covar  : noise-covariance for the hydra  observations (6x6)
        """
        self.t_filt = t
        self.x_filt = np.reshape(x_init, (12,1))
        self.S_filt = S_init
        self.motion_covar    = motion_covar


    def get_motion_covar(self, dt):
        """
        Returns the noise covariance for the motion model.
        Assumes a diagonal structure for now.
        """
        return self.motion_covar


    def get_motion_mat(self, dt):
        """
        Return the matrix A in x_t+1 = Ax_t + n_t.
        dt is the time-lapsed, i.e. the value of (t+1 - t).
        
        Return the matrix corresponding to a state of (x, vel_x, rot, vel_rot).
        """
        m = np.eye(6)
        m[0:3, 3:6] = dt*np.eye(3)
        return scl.block_diag(m,m)


    def get_motion_mats(self, dt):
        return (self.get_motion_mat(dt), self.get_motion_covar(dt))


    def get_obs_mat(self):
        """
        Return the observation matrix ('C')
        """
        return self.obs_mat


    def control_update(self, x_p, S_p, dt=None):
        """
        Runs the motion model forward by time dt, assuming previous mean is x_p
        and previous covariance is S_p.
        Returns the next mean and covariance (x_n, S_n).
        """
        if dt == None:
            dt = 1./self.freq

        A, R = self.get_motion_mats(dt)
        x_n = A.dot(x_p)
        S_n = A.dot(S_p).dot(A.T) + R
        x_n[6:9] = put_in_range(x_n[6:9])
        self.t_filt += dt
        return (x_n, S_n)


    def measurement_update(self, z_obs, C_obs, Q_obs, x_b, S_b):
        """
        z_obs        : Measurement vector
        C_obs, Q_obs : Measurement matrix and measurement noise covariance.
        x_b, S_b     : control-prediction of mean and covariance.
        
        returns the updated mean x_n and the covariance S_n.
        """
        x_b = np.reshape(x_b, (12,1))
        L = np.linalg.inv(C_obs.dot(S_b).dot(C_obs.T) + Q_obs)
        K = S_b.dot(C_obs.T).dot(L)
        x_n = x_b + K.dot(z_obs - C_obs.dot(x_b))
        S_n = S_b - K.dot(C_obs).dot(S_b)
        
        x_n[6:9] = put_in_range(x_n[6:9])
        return (x_n, S_n)


    def register_tf_observation(self, obs_tf, is_hydra, Q_obs, t=None, do_control_update=False):
        """
        This function updates the filter with an observation.
        OBS_TF   : is the observed transform.
        Q_OBS    : Noise covariance for this observation.
        DO_CONTROL_UPDATE : if true, a control update is performed before the observation update.
        """
        if do_control_update:
            if t == None:
                dt = 1./self.freq
            else:
                dt = t - self.t_filt
            self.x_filt, self.S_filt = self.control_update(self.x_filt, self.S_filt, dt)

        if obs_tf==None:
            return

        pos, rpy = self.canonicalize_obs(obs_tf)
        C_obs = self.get_obs_mat()
        z_obs = np.c_['0,2', pos, rpy]
        self.x_filt, self.S_filt = self.measurement_update(z_obs, C_obs, Q_obs, self.x_filt, self.S_filt)

