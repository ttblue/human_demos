from __future__ import division
import hd_utils.transformations as tfm
from hd_utils.colorize import colorize
import numpy as np
import scipy.linalg as scl

from collections import deque


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
        self.ar1_x_std     = 0.05 # m / sample
        self.ar1_r_std     = 5    # deg / sample
        self.ar2_x_std     = 0.05 # m / sample
        self.ar2_r_std     = 5    # deg / sample
        self.hydra_vx_std = 0.01 # m/s / sample
        self.hydra_r_std  = 0.1  # deg/ sample

        ## convert the above numbers to radians:
        self.min_r_std   = put_in_range(np.deg2rad(self.min_r_std))
        self.min_vr_std  = put_in_range(np.deg2rad(self.min_vr_std))
        self.r_std_t     = put_in_range(np.deg2rad(self.r_std_t))
        self.vr_std_t    = put_in_range(np.deg2rad(self.vr_std_t))
        self.ar1_r_std    = put_in_range(np.deg2rad(self.ar1_r_std))
        self.ar2_r_std    = put_in_range(np.deg2rad(self.ar2_r_std))
        self.hydra_r_std = put_in_range(np.deg2rad(self.hydra_r_std))

        # update frequency : the kalman filter updates the estimate explicitly at this rate.
        # it also updates when a measurement is given to it.
        self.freq = 30.
    
        ## last observations : used to calculate observation velocities.
        self.hydra_prev = None
        self.ar2_prev    = None
        self.ar2_prev   = None

        ## the filter's current belief and its time:
        self.t_filt = None # time
        self.x_filt = None # mean
        self.S_filt = None # covariance
 
        self.motion_covar = None
        self.hydra_covar = None
        self.ar1_covar = None
        self.ar2_covar = None

        ## store the observation matrix for the hydras and AR markers:
        ##  both hydra and ar markers observe xyz and rpy only:
        self.hydra_mat = np.zeros((6,12))
        self.hydra_mat[0:3, 0:3] = np.eye(3)
        self.hydra_mat[3:6, 6:9] = np.eye(3)

        self.hydra_vmat = np.zeros((6,12))
        self.hydra_vmat[0:3, 3:6] = np.eye(3)
        self.hydra_vmat[3:6, 6:9] = np.eye(3)

        self.ar1_mat = self.hydra_mat
        self.ar2_mat = self.hydra_mat
        
        self.hydra_ar1_system_err = deque([], 10)
        self.hydra_ar2_system_err = deque([], 10)
        
        self.id = 0



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
        
    def get_hydra_vmats(self):
        return (self.hydra_vmat, self.hydra_covar_vel) # covariance of hydra_v is also hydra_covar??

    def get_ar1_mats(self):
        """
        Returns a tuple : observation matrix and the corresponding noise covariance matrix
                          for AR marker observations from camera 1.
        AR markers observe the translation to very high accuracy, but rotations are very noisy. 
        """
        return (self.ar1_mat, self.ar1_covar)

    def get_ar2_mats(self):
        """
        Returns a tuple : observation matrix and the corresponding noise covariance matrix
                          for AR marker observations from camera 2.
        AR markers observe the translation to very high accuracy, but rotations are very noisy. 
        """
        return (self.ar2_mat, self.ar2_covar)
        

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
    

    def init_filter(self, t, x_init, S_init, motion_covar, hydra_covar, hydra_covar_vel, ar1_covar, ar2_covar):
        """
        Give the initial estimate for the filter.
        t is the time of the estimate.
        x_init is a 12 dimensional state vector.
        S_init is the state covariance (12x12).
        """
        self.t_filt = t
        self.x_filt = np.reshape(x_init, (12,1))
        self.S_filt = S_init
        self.motion_covar    = motion_covar
        self.hydra_covar     = hydra_covar
        self.hydra_covar_vel = hydra_covar_vel
        self.ar1_covar       = ar1_covar
        self.ar2_covar       = ar2_covar
        self.qcount = 0

    def __check_time__(self, t):
        if self.t_filt == None:
            print colorize('[Filter ERROR:] Filter not initialized.', 'red', True)
            return False
            
        if self.t_filt >= t:
            print colorize('[Filter ERROR:] Observation behind the filter estimate in time. Ignoring update.', 'blue', True)
            return False
        
        return True

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


#     def observe_ar(self, T_obs, t):
#         """
#         Update the filter to incorporate the observation from AR marker.
#         The marker transformation T_OBS is the estimate at time T.
#         
#         AR markers observe the translation to very high accuracy,
#         but rotations are very noisy. 
#         """
#         if not self.__check_time__(t):
#             return
#         
#         dt = t - self.t_filt
#         pos, rpy = self.canonicalize_obs(T_obs)
#         
#         z_obs = np.c_['0,2', pos, rpy]
#         
#         # update the last AR marker estimate 
#         self.ar_prev = (t, z_obs)
#         
#         C,Q = self.get_ar_mats()
#         self.x_filt, self.S_filt = self.control_update(self.x_filt, self.S_filt, dt)
#         self.x_filt, self.S_filt = self.measurement_update(z_obs, C, Q, self.x_filt, self.S_filt)
#        
# 
#     def observe_hydra(self, T_obs, t):
#         """
#         Update the filter to incorporate the observation from hydra.
#         The marker transformation T_OBS is the estimate at time T.
#         
#         Hydras observe rotations and the translation velocities to high accuracy.    
#         """
#         if not self.__check_time__(t):
#             return
#         
#         dt = t - self.t_filt
#         pos, rpy = self.canonicalize_obs(T_obs)
#         z_obs = np.c_['0,2', pos, rpy]           
# 
#         if self.hydra_prev == None:   ## in this case, we cannot observe the translational velocity
#             self.hydra_prev = (t, z_obs)
#             return
# 
#         ## calculate translation velocity : NOT BEING USED CURRENTLY
#         t_p, z_b = self.hydra_prev
#         vx_obs   = (z_obs[0:3] - z_b[0:3]) / (t-t_p)
#         self.hydra_prev = (t, z_obs)
# 
#         '''
#         for hydra still xyz and rpy
#         '''
#         z = z_obs#np.c_['0,2', vx_obs, rpy]            
#         C,Q = self.get_hydra_mats()
#         self.x_filt, self.S_filt = self.control_update(self.x_filt, self.S_filt, dt)
#         self.x_filt, self.S_filt = self.measurement_update(z, C, Q, self.x_filt, self.S_filt)


    def estimate_system_error(self, t):
        diff1_ready = False
        diff2_ready = False
        
        hydra_ar1_system_err = list(self.hydra_ar1_system_err)
        hydra_ar2_system_err = list(self.hydra_ar2_system_err)
        
        if len(hydra_ar1_system_err) == 10:
            diff1_ready = True
        if len(hydra_ar2_system_err) == 10:
            diff2_ready = True
            
        diff1_mean = np.zeros(3)
        diff1_std = np.zeros(3)
        diff2_mean = np.zeros(3)
        diff2_std = np.zeros(3)
        
        if diff1_ready:
            diff1_mean = np.mean(hydra_ar1_system_err, 0)
            diff1_std = np.std(hydra_ar1_system_err, 0)
            
        if diff2_ready:
            diff2_mean = np.mean(hydra_ar2_system_err, 0)
            diff2_std = np.std(hydra_ar2_system_err, 0)
        
        
        return (diff1_ready, diff1_mean, diff1_std, 
                diff2_ready, diff2_mean, diff2_std)
        
        

    def register_observation(self, t, T_ar1=None, T_ar2=None, T_hy=None):
        self.id += 1
        """
        New interface function to update the filter
        with observations from hydra and two kinects.
        
        Can pass in any combination of the hydra/camera1-ar-marker/camera2-ar-marker estimate.
        t is the time of the observation.
        
        NOTE: This does not update the {ar1, ar2, hydra}_prev variables:
              THIS WILL CAUSE ERRORS if using velocities in observation updates.
              ======================
        """
        if not self.__check_time__(t):
            return

        dt = t - self.t_filt
        self.x_filt, self.S_filt = self.control_update(self.x_filt, self.S_filt, dt)
        
        z_obs, C, Q = None, None, None
        AR_reading = False
        if T_hy != None or T_ar1 != None or T_ar2 != None: # initialize if anything was observed 
            z_obs = np.array([])
            C = None
            Q = None
            
        # remove outlier according to system error bound
        err1_ready, err1_mean, err1_std, err2_ready, err2_mean, err2_std = self.estimate_system_error(t)
        
        T_ar1_outlier = False
        T_ar2_outlier = False
        
        
        if T_hy != None and T_ar1 != None:
            if err1_ready:
                err = T_ar1[0:3, 3] - T_hy[0:3, 3] + err1_mean                
                cmp_res = np.greater(np.abs(err), np.array([0.02, 0.05, 0.02]))
                if cmp_res[0] == True or cmp_res[1] == True or cmp_res[2] == True:
                    print err, err1_std
                    T_ar1_outlier = True
 
                 
        if T_hy != None and T_ar2 != None:
            if err2_ready:
                err = T_ar2[0:3, 3] - T_hy[0:3, 3] + err2_mean  
                cmp_res = np.greater(np.abs(err), np.array([0.02, 0.05, 0.02]))
                if cmp_res[0] == True or cmp_res[1] == True or cmp_res[2] == True:
                    print err, err2_std
                    T_ar2_outlier = True
        

        # update system error estimation
        if T_hy != None and T_ar1 != None and T_ar1_outlier == False:
            pos_ar1 = T_ar1[0:3, 3]
            pos_hy = T_hy[0:3, 3]
            self.hydra_ar1_system_err.append(pos_hy - pos_ar1)
        
        if T_hy != None and T_ar2 != None and T_ar2_outlier == False:
            pos_ar2 = T_ar2[0:3, 3]
            pos_hy = T_hy[0:3, 3]
            self.hydra_ar2_system_err.append(pos_hy - pos_ar2)
                        

        if T_ar1 != None and T_ar1_outlier == False: # observe the ar from camera 1
            pos, rpy     = self.canonicalize_obs(T_ar1)
            c_ar1, q_ar1 = self.get_ar1_mats()
            z_obs = np.c_['0,2', z_obs, pos]#, rpy]
            C     = c_ar1[0:3,:]
            Q     = q_ar1[0:3,0:3]
            AR_reading = True

        if T_ar2 != None and T_ar2_outlier == False: # observe the ar from camera 2
            pos, rpy     = self.canonicalize_obs(T_ar2)
            c_ar2, q_ar2 = self.get_ar2_mats()
            z_obs = np.c_['0,2', z_obs, pos]#, rpy]
            if not AR_reading:
                C = c_ar2[0:3,:]
                Q = q_ar2[0:3,0:3]
                AR_reading = True
            else:
                C = np.r_[C, c_ar2[0:3,:]]
                Q = scl.block_diag(Q, q_ar2[0:3,0:3])

        if T_hy != None and T_ar1 is None and T_ar2 is None:
        #if T_hy != None:# observe the hydra
            pos, rpy = self.canonicalize_obs(T_hy)

            c_hy, q_hy = self.get_hydra_mats()
            vc_hy, vq_hy = self.get_hydra_vmats()
            
#             if T_ar1 is None and T_ar2 is None:
#                 if err1_ready and err2_ready: # if system err is estimated
#                     z_obs = np.c_['0, 2', z_obs, pos - (err1 + err2) * 0.5]
#                 elif err1_ready:
#                     z_obs = np.c_['0, 2', z_obs, pos - err1]
#                 elif err2_ready:
#                     z_obs = np.c_['0, 2', z_obs, pos - err2]
#                 else:
#                     z_obs = np.c_['0, 2', z_obs, pos]
#          
#                 if not AR_reading:
#                     C = c_hy[0:3,:]
#                     Q = q_hy[0:3, 0:3]
#                     AR_reading = True
#                 else:
#                     C = np.r_[C, c_hy[0:3,:]]
#                     Q = scl.block_diag(Q, q_hy[0:3, 0:3])               
            
            z_obs = np.c_['0,2', z_obs, rpy]
            if not AR_reading:
                C = c_hy[3:6,:]
                Q = q_hy[3:6,3:6]
                AR_reading = True
            else:
                C = np.r_[C, c_hy[3:6,:]]
                Q = scl.block_diag(Q, q_hy[3:6,3:6])

            
            #if T_ar1 is None and T_ar2 is None and self.hydra_prev != None:
            if self.hydra_prev != None:
                vpos = (pos - self.hydra_prev) / dt
                
                #change here
                z_obs = np.c_['0,2', z_obs, vpos]
                C = np.r_[C, vc_hy[0:3,:]]
                Q = scl.block_diag(Q, vq_hy[0:3,0:3])            
                
            self.hydra_prev = pos
        else:
            self.hydra_prev = None

        if (z_obs != None and C!= None and Q!=None):
            self.x_filt, self.S_filt = self.measurement_update(z_obs, C, Q, self.x_filt, self.S_filt)




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
