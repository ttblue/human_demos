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
        self.x_std_t  = 0.001  # m/s
        self.vx_std_t = 1  # m/s /s
        self.r_std_t  = 40 # deg/s
        self.vr_std_t = 90 # deg/s /s
      
        ## standard deviations of the measurements:
        self.ar_x_std     = 0.05 # m / sample
        self.ar_r_std     = 5    # deg / sample
        self.hydra_vx_std = 0.01 # m/s / sample
        self.hydra_r_std  = 0.1  # deg/ sample

        ## convert the above numbers to radians:
        self.min_r_std   = self.put_in_range(np.deg2rad(self.min_r_std))
        self.min_vr_std  = self.put_in_range(np.deg2rad(self.min_vr_std))
        self.r_std_t     = self.put_in_range(np.deg2rad(self.r_std_t))
        self.vr_std_t    = self.put_in_range(np.deg2rad(self.vr_std_t))
        self.ar_r_std    = self.put_in_range(np.deg2rad(self.ar_r_std))
        self.hydra_r_std = self.put_in_range(np.deg2rad(self.hydra_r_std))

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

        self.motion_covar = None
        self.hydra_covar = None
        self.ar_covar = None

        ## store the observation matrix for the hydras and AR markers:
        ##  both hydra and ar markers observe xyz and rpy only:
        self.hydra_mat = np.zeros((6,12))
        self.hydra_mat[0:3, 0:3] = np.eye(3)
        self.hydra_mat[3:6, 6:9] = np.eye(3)

        self.ar_mat = self.hydra_mat


    def get_motion_covar(self, dt=1./30.):
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


    def get_hydra_mats(self):
        """
        Returns a tuple : 1. The observation matrix mapping the state to the observation.
                          2. The noise covariance matrix

        Hydras observe rotations and the translation velocities to high accuracy.
        """
        return (self.hydra_mat, self.hydra_covar)
        

    def get_ar_mats(self):
        """
        Returns a tuple : observation matrix and the corresponding noise covariance matrix
                          for AR marker observations.
        AR markers observe the translation to very high accuracy, but rotations are very noisy. 
        """
        return (self.ar_mat, self.ar_covar)
        

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
        x_n[6:9] = self.put_in_range(x_n[6:9])
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
        
        x_n[6:9] = self.put_in_range(x_n[6:9])

        return (x_n, S_n)
    

    def init_filter(self, t, x_init, S_init, motion_covar, hydra_covar, ar_covar):
        """
        Give the initial estimate for the filter.
        t is the time of the estimate.
        x_init is a 12 dimensional state vector.
        S_init is the state covariance (12x12).
        """
        self.t_filt = t
        self.x_filt = np.reshape(x_init, (12,1))
        self.S_filt = S_init
        self.motion_covar = motion_covar
        self.hydra_covar  = hydra_covar
        self.ar_covar     = ar_covar
        

    def __check_time__(self, t):
        if self.t_filt == None:
            print colorize('[Filter ERROR:] Filter not initialized.', 'red', True)
            return False
            
        if self.t_filt >= t:
            print colorize('[Filter ERROR:] Observation behind the filter estimate in time. Ignoring update.', 'blue', True)
            return False
        
        return True


    def put_in_range(self, x):
        """
        Puts all the values in x in the range [-180, 180).
        """
        return ( (x+np.pi)%(2*np.pi) )- np.pi
            

    def closer_angle(self, x, a):
        """
        returns the angle f(x) which is closer to the angle a in absolute value.
        """
        return a + self.put_in_range(x-a)  


    def canonicalize_obs(self, T_obs):
        """
        Returns the position and translation from T_obs (4x4 mat).
        Puts the rotation in a form which makes it closer to filter's
        current estimate in absolute terms.
        """
        pos = T_obs[0:3,3]
        rpy = np.array(tfm.euler_from_matrix(T_obs), ndmin=2).T
        rpy = self.closer_angle(rpy, self.x_filt[6:9])
        return (pos, rpy)


    def observe_ar(self, T_obs, t):
        """
        Update the filter to incorporate the observation from AR marker.
        The marker transformation T_OBS is the estimate at time T.
        
        AR markers observe the translation to very high accuracy,
        but rotations are very noisy. 
        """
        if not self.__check_time__(t):
            return
        
        dt = t - self.t_filt
        pos, rpy = self.canonicalize_obs(T_obs)
        
        z_obs = np.c_['0,2', pos, rpy]
        
        # update the last AR marker estimate 
        self.ar_prev = (t, z_obs)
        
        C,Q = self.get_ar_mats()
        self.x_filt, self.S_filt = self.control_update(self.x_filt, self.S_filt, dt)
        self.x_filt, self.S_filt = self.measurement_update(z_obs, C, Q, self.x_filt, self.S_filt)
       

    def observe_hydra(self, T_obs, t):
        """
        Update the filter to incorporate the observation from hydra.
        The marker transformation T_OBS is the estimate at time T.
        
        Hydras observe rotations and the translation velocities to high accuracy.    
        """
        if not self.__check_time__(t):
            return
        
        dt = t - self.t_filt
        pos, rpy = self.canonicalize_obs(T_obs)
        z_obs = np.c_['0,2', pos, rpy]           

        if self.hydra_prev == None:   ## in this case, we cannot observe the translational velocity
            self.hydra_prev = (t, z_obs)
            return

        ## calculate translation velocity : NOT BEING USED CURRENTLY
        t_p, z_b = self.hydra_prev
        vx_obs   = (z_obs[0:3] - z_b[0:3]) / (t-t_p)
        self.hydra_prev = (t, z_obs)

        z = z_obs#np.c_['0,2', vx_obs, rpy]            
        C,Q = self.get_hydra_mats()
        self.x_filt, self.S_filt = self.control_update(self.x_filt, self.S_filt, dt)
        self.x_filt, self.S_filt = self.measurement_update(z, C, Q, self.x_filt, self.S_filt)


    def register_observation(self, t, T_ar=None, T_hy=None):
        """
        New interface function to update the filter
        with observations.
        
        Can pass in both or any one of the hydra/ ar-marker estimate.
        t is the time of the observation.
        
        NOTE: This does not update the {ar, hydra}_prev variables:
              THIS WILL CAUSE ERRORS if using velocities in observation updates.
              ======================
        """ 
        if not self.__check_time__(t):
            return
        
        dt = t - self.t_filt
        self.x_filt, self.S_filt = self.control_update(self.x_filt, self.S_filt, dt)
       
        # observe just the hydras:
        z_obs, C, Q = None, None, None
        if T_ar==None and T_hy != None:
            pos, rpy = self.canonicalize_obs(T_hy)
            z_obs = np.c_['0,2', pos, rpy]
            C,Q = self.get_hydra_mats()
        elif T_ar!=None and T_hy == None: # observe just the ar:
            pos, rpy = self.canonicalize_obs(T_ar)
            z_obs = np.c_['0,2', pos, rpy]
            C,Q = self.get_ar_mats()
        elif T_ar != None and T_hy != None: # observe both
            p1, th1 = self.canonicalize_obs(T_ar)
            C1,Q1   = self.get_ar_mats()
            p2, th2 = self.canonicalize_obs(T_hy)
            C2,Q2   = self.get_hydra_mats()

            z_obs = np.c_['0,2', p1, th1, p2, th2]
            C = np.r_[C1, C2]
            Q = scl.block_diag(Q1, Q2)

        if (z_obs != None and C!=None and Q!=None):
            self.x_filt, self.S_filt = self.measurement_update(z_obs, C, Q, self.x_filt, self.S_filt)


