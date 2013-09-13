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
import matplotlib.pylab as plt

hd_path = '/home/henrylu/henry_sandbox/human_demos'


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


def fit_process_noise(fname=None, f=30.):
    """
    Gets the motion-model noise covariance.
    fname : file name of pickle file containing joint-angles of the pr2.
    f     : the frequency at which the joints were logged.
    
    covariance is calculated as:
        covar = 1/n sum_i=1^n (x(t+1) - A*x(t)).T * (x(t+1) - A*x(t)) 
    
    """
    dt = 1.0/f   
    if fname==None:
        fname = hd_path + '/hd_track/data/pr2-knot-tie-joint-states.cpickle'

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


def fit_hydra_noise(Ts_bg, Ts_bh, T_gh, f):
    """
    Get the hydra measurement covariance.
    Ts_bg : list of transforms from pr2's base to its gripper holding the hydra.
    Ts_bh : list of transforms from pr2's base to hydra sensor.
    T_gh :  One offset transform from pr2's gripper holding the hydra and the hydra sensor.
    f    : the frequency at which the data is logged. 
    """
    dt = 1./f
    
    assert len(Ts_bg) == len(Ts_bh), "Number of hydra and pr2 transforms not equal."
    Ts_bg_gh = [t.dot(T_gh) for t in Ts_bg]

    ## extract the full state vector:    
    X_bh    = state_from_tfms(Ts_bh, dt).T
    X_bg_gh = state_from_tfms(Ts_bg_gh, dt).T
    X_bg_gh[6:9,:] = closer_angle(X_bg_gh[6:9,:], X_bh[6:9,:])
    
    C = kalman().get_hydra_mats()[0]
    
    err = C.dot(X_bh) - C.dot(X_bg_gh)
    covar = (err.dot(err.T))/err.shape[1]
    return (err, covar)

    
def fit_ar_noise(Ts_bg, Ts_ba_raw, T_ga, f):
    """
    Similar to the function above for fitting hydra-noise covariance. 
    """
    dt = 1./f
 
    assert len(Ts_bg) == len(Ts_ba_raw), "Number of ar and pr2 transforms not equal."
    Ts_ba = []
    Ts_bg_ga = []
    for i in xrange(len(Ts_ba_raw)):
        T_ba = Ts_ba_raw[i]
        T_bg = Ts_bg[i]
        if T_ba == None:
            continue
        else:
            #print len(t)
            Ts_bg_ga.append(T_bg.dot(T_ga))
            Ts_ba.append(T_ba)
    assert len(Ts_bg_ga) == len(Ts_ba), "Number of valid ar and pr2 transforms not equal."

    ## extract the full state vector:    
    X_ba    = state_from_tfms(Ts_ba, dt).T
    X_bg_ga = state_from_tfms(Ts_bg_ga, dt).T
    X_bg_ga[6:9,:] = closer_angle(X_bg_ga[6:9,:], X_ba[6:9,:])

    C = kalman().get_ar_mats()[0]
    
    err = C.dot(X_ba) - C.dot(X_bg_ga)
    covar = (err.dot(err.T))/err.shape[1]
    return (err, covar)
    


def plot_ar_data(Ts_bg, Ts_ba, T_ga, f):
    """
    Plot the data from the kinect and the pr2.
    """
    dt = 1./f

    assert len(Ts_bg) == len(Ts_ba), "Number of ar and pr2 transforms not equal."
    Ts_ba_ag = []
    for t in Ts_ba:
        if t == None:
            Ts_ba_ag.append(np.zeros((4,4)))
        else:
            #print len(t)
            Ts_ba_ag.append(t.dot(np.linalg.inv(T_ga)))

    ## plot the translation:
    axlabels = ['x','y','z']
    for i in range(3):
        plt.subplot(3,2,i+1)
        plt.plot(np.array([t[i,3] for t in Ts_bg]), label='pr2')
        plt.plot(np.array([t[i,3] for t in Ts_ba_ag]),'.', label='kinect')
        plt.ylabel(axlabels[i])
        plt.legend()
    plt.show()

def plot_and_fit_ar(plot=True, f=30.):
    dic = cPickle.load(open(hd_path + '/hd_track/data/pr2-hydra-kinect-trajectory-transforms.cpickle'))
    Ts_bg = dic['Ts_bg']
    Ts_ba = dic['Ts_bk']
    #rint len(Ts_bk)
    T_ga = dic['T_gk']

    if plot:
        plot_ar_data(Ts_bg, Ts_ba, T_ga, f)

    return fit_ar_noise(Ts_bg, Ts_ba, T_ga, f)

def plot_hydra_data(Ts_bg, Ts_bh, T_gh, f):
    """
    Plot the data from the hydra and the pr2.
    """
    dt = 1./f

    assert len(Ts_bg) == len(Ts_bh), "Number of hydra and pr2 transforms not equal."
    Ts_bg_gh = [t.dot(T_gh) for t in Ts_bg]
    
    ## plot the translation:
    axlabels = ['x','y','z']
    for i in range(3):
        plt.subplot(3,2,i+1)
        plt.plot(np.array([t[i,3] for t in Ts_bh]), label='hydra')
        plt.plot(np.array([t[i,3] for t in Ts_bg_gh]), label='pr2')
        plt.ylabel(axlabels[i])
        plt.legend()
    
    ## plot the rotation:
    X_bh    = state_from_tfms(Ts_bh, dt).T
    X_bg_gh = state_from_tfms(Ts_bg_gh, dt).T
    X_bg_gh[6:9,:] = closer_angle(X_bg_gh[6:9,:], X_bh[6:9,:])

    axlabels = ['roll','pitch','yaw']
    for i in range(3):
        plt.subplot(3,2,i+4)      
        plt.plot(X_bh[i+6,:], label='hydra')
        plt.plot(X_bg_gh[i+6,:], label='pr2')
        plt.ylabel(axlabels[i])
        plt.legend()

    plt.show()


def plot_and_fit_hydra(plot=True, f=30.):

    dic = cPickle.load(open(hd_path + '/hd_track/data/pr2-hydra-kinect-trajectory-transforms.cpickle'))
    Ts_bg = dic['Ts_bg']
    Ts_bh = dic['Ts_bh']
    #rint len(Ts_bk)
    T_gh = dic['T_gh']

    if plot:
        plot_hydra_data(Ts_bg, Ts_bh, T_gh, f)

    return fit_hydra_noise(Ts_bg, Ts_bh, T_gh, f)


def save_kalman_covars(out_file='./data/pr2-hydra-kinect-covars-xyz-rpy.cpickle'):
    """
    Computes the process noise covariance and the hydra-measurement noise covariances
    from data and saves them to a cpickle file.
    """
    le,lc,re,rc = fit_process_noise()
    he, hc = plot_and_fit_hydra(False)
    ae, ac = plot_and_fit_ar(False)
    
    dict = {'process': (lc+rc)/2, 'hydra':hc, 'kinect':ac}
    cPickle.dump(dict, open(out_file, 'wb'))


if __name__ == '__main__':
    pass
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--fname", help="joints file name.", required=True)
    #args = parser.parse_args()
    
    #le, lc, re, rc = fit_process_noise(args.fname)    
    #print "LEFT COVAR : ", lc
    #print "RIGHT COVAR : ", rc
