from __future__ import division
from kalman import kalman
import hd_utils.transformations as tfms
from hd_visualization.mayavi_plotter import *
from hd_visualization.mayavi_utils import *
import numpy as np


def rotation_matrix(axis,theta):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def add_noise(Ts, rotn = 5, xn = 0.0):
    """
    Adds noise to each transform in the list of transforms.
    rotn : standard dev of roll,pitch,yaw noise
    xn   : standard dev of translation noise 
    """
    rotn = np.deg2rad(rotn)
    Tn = []
    for t in Ts:
        rpy = np.array(tfms.euler_from_matrix(t))
        rpy += rotn*np.random.randn(3)
        tn  = tfms.euler_matrix(rpy[0], rpy[1], rpy[2]) 
        tn[0:3,3] = t[0:3,3] + xn*np.random.randn(3)
        Tn.append(tn)
    return Tn


def gen_tfms(n=100, f=30):
    dt = 1./f
    
    rax  = np.array((1,0,0))
    v_rx  = np.deg2rad(180)
    
    v_x   = 1
    xax = np.array((1,1,0)) 

    ts = dt*np.arange(n)
    Ts = []
    
    x_init = np.zeros(3)
    r_init = np.zeros(3)
    
    for t in ts:
        x = x_init  + t*v_x*xax
        r = r_init  + t*v_rx*rax
        
        T = tfms.euler_matrix(r[0], r[1], r[2])
        T[0:3,3] = x
        Ts.append(T)

    return Ts


def x_to_tf(Xs):
    """
    Converts a list of 12 dof state vector to a list of transforms.
    """
    Ts = []
    for x in Xs:
        trans = x[0:3]
        rot   = x[6:9]
        T = tfms.euler_matrix(rot[0], rot[1], rot[2])
        T[0:3,3] = np.reshape(trans, 3)
        Ts.append(T)
    return Ts


if __name__ == '__main__':
    N = 10
    t_true  = gen_tfms(N)
    t_noise = add_noise(t_true)
    
    plotter = PlotterInit()
    
    for t in t_true:
        req  = gen_custom_request('transform', t, size=0.05)
        plotter.request(req)

    for t in t_noise:
        req  = gen_custom_request('transform', t, size=0.02)
        plotter.request(req)    
    true2noise = []
    for i in xrange(len(t_true)):
        true2noise.append(np.c_[t_true[i][0:3,3], t_noise[i][0:3,3]].T)
    lreq = gen_custom_request('lines', true2noise, color=(1,1,0), line_width=3)
    plotter.request(lreq)
        
    kf = kalman()
    kf.init_filter(0, np.zeros(12), 0.01*np.eye(12))
    ts = (1/30.) * np.arange(N)

    ke = []
    for i in xrange(1,len(ts)):
        kf.observe_ar(t_noise[i], ts[i])
        ke.append(kf.x_filt)
    kf_Ts = x_to_tf(ke)
    
    for t in kf_Ts:
        req  = gen_custom_request('transform', t, size=0.03)
        plotter.request(req)    
    true2est = []
    for i in xrange(len(t_true)-1):
        true2est.append(np.c_[t_true[i+1][0:3,3], kf_Ts[i][0:3,3]].T)
    lreq = gen_custom_request('lines', true2est, color=(0,1,1), line_width=3)
    plotter.request(lreq)
