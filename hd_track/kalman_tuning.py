"""
Gets the noise parameters for the kalman filter using ground-truth data.
"""
from __future__ import division
import cPickle
import openravepy as rave
import numpy as np
from hd_track.kalman import kalman
from hd_utils import transformations as tfms
import argparse

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


def tfms_from_joints(joints, side='l'):
    """
    returns a list of end-effector poses (gripper_toolframe)
    given a matrix of arm-joint values : nx7 for the pr2 robot.
    
    side \in {'l', 'r'} is  which arm to use.
    """
    Ts = []
    
    env = rave.Environment()
    env.Load('robots/pr2-beta-static.zae')
    pr2 = env.GetRobot('pr2')
    manip = pr2.GetManipulator('%sarm'% 'left' if 'l' in side else 'right')
    arm_joints = manip.GetArmIndices()
    
    for i in xrange(joints.shape[0]):
        pr2.SetJointValues(joints[i,:], arm_joints)
        Ts.append(manip.GetEndEffectorTransform())
        
    env.Destroy()
    del env
    return Ts


def state_from_tfms(Ts, dt=1./30.):
    """
    returns (n-1)x12 matrix of the states of kalman filter
    derived from the list 'Ts' of transforms logged at 
    frequency = 1/dt.
    The 'n' above is the number of transforms in the list 'Ts'.
    """
    N = len(Ts)
    Xs = np.empty((N-1, 12))

    for i in xrange(N-1):
        if i==0:
            pos_prev = Ts[0][0:3,3]
            rpy_prev = np.array(tfms.euler_from_matrix(Ts[0]))
        else:
            pos_prev = Xs[i-1,0:3]
            rpy_prev = Xs[i-1,6:9]

        Xs[i,0:3] = Ts[i+1][0:3,3]
        Xs[i,3:6] = (Xs[i,0:3] - pos_prev)/dt
        Xs[i,6:9] = np.array(tfms.euler_from_matrix(Ts[i+1]))       
        Xs[i,9:12] = (closer_angle(Xs[i,6:9], rpy_prev) - rpy_prev)/dt 

    return Xs

def fit_process_noise(fname, f=30.):
    """
    Gets the motion-model noise covariance.
    fname : file name of pickle file containing joint-angles of the pr2.
    f     : the frequency at which the joints were logged.
    
    covariance is calculated as:
        covar = 1/n sum_i=1^n (x(t+1) - A*x(t)).T * (x(t+1) - A*x(t)) 
    
    """
    dt = 1.0/f
    
    joints = cPickle.load(open(fname, 'rb'))['mat']
    rstates = state_from_tfms(tfms_from_joints(joints[:,0:7])).T
    lstates = state_from_tfms(tfms_from_joints(joints[:,8:15])).T

    A = kalman().get_motion_mat(dt)

    r_err = A.dot(rstates[:,0:-1]) - rstates[:,1:]
    r_err[6:9,:]  = put_in_range(r_err[6:9,:])
    r_err = r_err[:,np.nonzero(np.sum(r_err, axis=0))[0]] # take out all zero columns : not sure if correct thing to do.
    r_covar = r_err.dot(r_err.T) / r_err.shape[1]
    
    l_err = A.dot(lstates[:,0:-1]) - lstates[:,1:]
    l_err[6:9,:]  = put_in_range(l_err[6:9,:])
    l_err = l_err[:, np.nonzero(np.sum(l_err, axis=0))[0]] 
    l_covar = l_err.dot(l_err.T) / l_err.shape[1]

    return (l_err, l_covar, r_err, r_covar)


def fit_hydra_noise(T_bg, T_bh, Tf=30.):
    dt = 1./f
    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fname", help="joints file name.", required=True)
    args = parser.parse_args()
    
    le, lc, re, rc = fit_process_noise(args.fname)    
    print "LEFT COVAR : ", lc
    print "RIGHT COVAR : ", rc
