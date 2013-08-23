import networkx as nx
import numpy as np, numpy.linalg as nlg

from hd_calib import gripper_calibration as gc

def rotation_matrix(axis,theta):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    

def make_obs(trans_SCALE=0.1, theta_SCALE=1.5):
    axis = np.random.randn(3)
    theta = np.random.rand()*np.pi*theta_SCALE
    
    t = np.random.randn(3)*trans_SCALE
    
    tfm = np.eye(4)
    tfm[0:3,0:3] = rotation_matrix(axis, theta)
    tfm[0:3,3] = t
    
    return tfm

def test_gc (n):
    
    G = nx.Graph()
    
    T12 = make_obs()
    T23 = make_obs()
    
    for _ in xrange(n):
        T1 = make_obs()
        noise1 = make_obs(0.01,0.02)
        T2 = T1.dot(T12).dot(noise1)
        noise2 = make_obs(0.01,0.02)
        T3 = T2.dot(T23).dot(noise2)
        
        tfms = {1:T1,2:T2,3:T3}
        
        gc.update_graph_from_observations(G, tfms)
        
    G_opt = gc.compute_relative_transforms (G, 3, n)
    
    print "T12"
    print "Original:"
    print T12
    print "Calculated:"
    print G_opt.edge[1][2]['tfm'], '\n'
    
    print "T23"
    print "Original:"
    print T23
    print "Calculated:"
    print G_opt.edge[2][3]['tfm'], '\n'
    
    print "T31"
    print "Original:"
    print nlg.inv(T12.dot(T23))
    print "Calculated:"
    print G_opt.edge[3][1]['tfm'], '\n'