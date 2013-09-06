#!/usr/bin/python
import networkx as nx
import numpy as np, numpy.linalg as nlg

from hd_calib import gripper_calibration as gc, cyni_cameras as cc
from hd_utils import utils

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
    

def test_gc2 (n, add_noise=False):
    ##
    # Markers 1,2,3 are on l finger
    # Markers 4,5,6 are on r finger
    # 7 and 8 are on the middle
    ##
    
    ### Initializing everything
    # create calib info
    calib_info = {"master": {"ar_markers":[7,8],
                              "master_group":True,
                              "angle_scale":0},
                  "l": {"ar_markers":[1,2,3],
                        "angle_scale":1},
                  "r": {"ar_markers":[4,5,6],
                        "angle_scale":-1}}
    
    masterGraph = nx.DiGraph()
    
    # Setup the groups
    for group in calib_info:
        masterGraph.add_node(group)
        masterGraph.node[group]["graph"] = nx.Graph()
        masterGraph.node[group]["angle_scale"] = calib_info[group]['angle_scale']
        
        if calib_info[group].get("ar_markers") is None:
            calib_info[group]["ar_markers"] = []
        if calib_info[group].get("hydras") is None:
            calib_info[group]["hydras"] = []
        if calib_info[group].get("ps_markers") is None:
            calib_info[group]["ps_markers"] = []
        
        if calib_info[group].get("master_group") is not None:
            masterGraph.node[group]["master"] = 1
            masterGraph.node[group]['angle_scale'] = 0

        masterGraph.node[group]["markers"] = calib_info[group]["ar_markers"] + calib_info[group]["hydras"] + calib_info[group]["ps_markers"]
        masterGraph.node[group]["calib_info"] = calib_info[group]
    ### Done initializing.
    
    T12 = make_obs()
    T23 = make_obs()
    
    T45 = make_obs()
    T56 = make_obs()
    
    T78 = make_obs()
    
    T7cor = make_obs()
    Tcor1 = make_obs()
    Tcor4 = make_obs()
    
    zaxis = np.array([0,0,1])
    
    for _ in range(n):
        theta = -15 + np.random.random()*30
        theta2 = utils.rad_angle(theta)

        T7 = make_obs()
        
        R1 = np.eye(4)
        R1[0:3,0:3] = rotation_matrix(zaxis,theta2)
        R4 = np.eye(4)
        R4[0:3,0:3] = rotation_matrix(zaxis,-theta2)
        
        if add_noise: 
            Tcor = T7.dot(T7cor)
            
            noise = make_obs(0.01,0.01)
            T1 = Tcor.dot(R1).dot(Tcor1).dot(noise)
            noise = make_obs(0.01,0.01)
            T2 = T1.dot(T12).dot(noise)
            noise = make_obs(0.01,0.01)
            T3 = T2.dot(T23).dot(noise)
            
            noise = make_obs(0.01,0.01)
            T4 = Tcor.dot(R4).dot(Tcor4).dot(noise)
            noise = make_obs(0.01,0.01)
            T5 = T4.dot(T45).dot(noise)
            noise = make_obs(0.01,0.01)
            T6 = T5.dot(T56).dot(noise)
            
            noise = make_obs(0.01,0.01)
            T8 = T7.dot(T78).dot(noise)
        else:
            noise = np.eye(4)
            
            Tcor = T7.dot(T7cor)
            
            T1 = Tcor.dot(R1).dot(Tcor1).dot(noise)
            T2 = T1.dot(T12).dot(noise)
            T3 = T2.dot(T23).dot(noise)
            
            T4 = Tcor.dot(R4).dot(Tcor4).dot(noise)
            T5 = T4.dot(T45).dot(noise)
            T6 = T5.dot(T56).dot(noise)
            
            T8 = T7.dot(T78).dot(noise)
            
        # Randomly sample a few until done maybe?
        tfms = {1:T1,2:T2,3:T3,4:T4,5:T5,6:T6,7:T7,8:T8}
        
        gc.update_groups_from_observations(masterGraph, tfms, theta)
            
    init = np.zeros(36)
    init[0:12] = nlg.inv(T78).dot(T7cor)[0:3,:].reshape(12,order='F')
    init[12:24] = nlg.inv(Tcor4)[0:3,:].reshape(12,order='F')
    init[24:36] = nlg.inv(Tcor1)[0:3,:].reshape(12,order='F')
    
    G_opt = gc.compute_relative_transforms(masterGraph, 5)#, init)
    
    
    # 7,8 markers
    group = G_opt.node["master"]
    G = group["graph"]
    print "T78"
    print "Original:"
    print T78
    print "Calculated:"
    print G.edge[7][8]['tfm'], '\n'
    
    # 1,2,3 markers
    group = G_opt.node["l"]
    G = group["graph"]
    print "T12"
    print "Original:"
    print T12
    print "Calculated:"
    print G.edge[1][2]['tfm'], '\n'
    
    print "T23"
    print "Original:"
    print T23
    print "Calculated:"
    print G.edge[2][3]['tfm'], '\n'
    
    print "T31"
    print "Original:"
    print nlg.inv(T12.dot(T23))
    print "Calculated:"
    print G.edge[3][1]['tfm'], '\n'
    
    # 4,5,6 markers
    group = G_opt.node["r"]
    G = group["graph"]
    print "T45"
    print "Original:"
    print T45
    print "Calculated:"
    print G.edge[4][5]['tfm'], '\n'
    
    print "T56"
    print "Original:"
    print T56
    print "Calculated:"
    print G.edge[5][6]['tfm'], '\n'
    
    print "T64"
    print "Original:"
    print nlg.inv(T45.dot(T56))
    print "Calculated:"
    print G.edge[6][4]['tfm'], '\n'
    
    # Let's try random angles:

    theta = -15 + np.random.random()*30
    theta2 = utils.rad_angle(theta)
    R1 = np.eye(4)
    R1[0:3,0:3] = rotation_matrix(zaxis,theta2)
    print "T71"
    print "Original:"
    print T7cor.dot(R1).dot(Tcor1)
    print "Calculated:"
    print G_opt.edge["master"]["l"]['tfm_func'](7,1,theta), '\n'
    print "norm diff:", nlg.norm(G_opt.edge["master"]["l"]['tfm_func'](7,1,theta)-T7cor.dot(R1).dot(Tcor1)), '\n'
    
    print "Difference between cors"
    T7cor_calib = G_opt.node["master"]["graph"].edge[7]["cor"]["tfm"]
    print nlg.inv(T7cor).dot(T7cor_calib)


def test_gc3 (n, add_noise=False):
    ##
    # Markers 1,2,3 are on l finger
    # Markers 4,5,6 are on r finger
    # 7 and 8 are on the middle
    ##
    
    ### Initializing everything
    # create calib info
    calib_info = {"master": {"ar_markers":[7,8],
                              "master_group":True,
                              "angle_scale":0},
                  "l": {"ar_markers":[1,2,3],
                        "angle_scale":1},
                  "r": {"ar_markers":[4,5,6],
                        "angle_scale":-1}}
    gripper_calib = gc.gripper_calibrator(None, calib_info=calib_info)
    gripper_calib.initialize_calibration(fake_data=True)
        
    T12 = make_obs()
    T23 = make_obs()
    
    T45 = make_obs()
    T56 = make_obs()
    
    T78 = make_obs()
    
    T7cor = make_obs()
    Tcor1 = make_obs()
    Tcor4 = make_obs()
    
    zaxis = np.array([0,0,1])
    
    for _ in range(n):
        theta = -15 + np.random.random()*30
        theta2 = utils.rad_angle(theta)

        T7 = make_obs()
        
        R1 = np.eye(4)
        R1[0:3,0:3] = rotation_matrix(zaxis,theta2)
        R4 = np.eye(4)
        R4[0:3,0:3] = rotation_matrix(zaxis,-theta2)
        
        if add_noise: 
            Tcor = T7.dot(T7cor)
            
            noise = make_obs(0.01,0.01)
            T1 = Tcor.dot(R1).dot(Tcor1).dot(noise)
            noise = make_obs(0.01,0.01)
            T2 = T1.dot(T12).dot(noise)
            noise = make_obs(0.01,0.01)
            T3 = T2.dot(T23).dot(noise)
            
            noise = make_obs(0.01,0.01)
            T4 = Tcor.dot(R4).dot(Tcor4).dot(noise)
            noise = make_obs(0.01,0.01)
            T5 = T4.dot(T45).dot(noise)
            noise = make_obs(0.01,0.01)
            T6 = T5.dot(T56).dot(noise)
            
            noise = make_obs(0.01,0.01)
            T8 = T7.dot(T78).dot(noise)
        else:
            noise = np.eye(4)
            
            Tcor = T7.dot(T7cor)
            
            T1 = Tcor.dot(R1).dot(Tcor1).dot(noise)
            T2 = T1.dot(T12).dot(noise)
            T3 = T2.dot(T23).dot(noise)
            
            T4 = Tcor.dot(R4).dot(Tcor4).dot(noise)
            T5 = T4.dot(T45).dot(noise)
            T6 = T5.dot(T56).dot(noise)
            
            T8 = T7.dot(T78).dot(noise)
            
        # Randomly sample a few until done maybe?
        tfms = {1:T1,2:T2,3:T3,4:T4,5:T5,6:T6,7:T7,8:T8}
        
        gc.update_groups_from_observations(gripper_calib.masterGraph, tfms, theta)
            
    init = np.zeros(36)
    init[0:12] = nlg.inv(T78).dot(T7cor)[0:3,:].reshape(12,order='F')
    init[12:24] = nlg.inv(Tcor4)[0:3,:].reshape(12,order='F')
    init[24:36] = nlg.inv(Tcor1)[0:3,:].reshape(12,order='F')
    
    gripper_calib.finish_calibration()
    G_opt = gripper_calib.transform_graph 
    
    
    # 7,8 markers
    group = G_opt.node["master"]
    G = group["graph"]
    print "T78"
    print "Original:"
    print T78
    print "Calculated:"
    print G.edge[7][8]['tfm'], '\n'
    
    # 1,2,3 markers
    group = G_opt.node["l"]
    G = group["graph"]
    print "T12"
    print "Original:"
    print T12
    print "Calculated:"
    print G.edge[1][2]['tfm'], '\n'
    
    print "T23"
    print "Original:"
    print T23
    print "Calculated:"
    print G.edge[2][3]['tfm'], '\n'
    
    print "T31"
    print "Original:"
    print nlg.inv(T12.dot(T23))
    print "Calculated:"
    print G.edge[3][1]['tfm'], '\n'
    
    # 4,5,6 markers
    group = G_opt.node["r"]
    G = group["graph"]
    print "T45"
    print "Original:"
    print T45
    print "Calculated:"
    print G.edge[4][5]['tfm'], '\n'
    
    print "T56"
    print "Original:"
    print T56
    print "Calculated:"
    print G.edge[5][6]['tfm'], '\n'
    
    print "T64"
    print "Original:"
    print nlg.inv(T45.dot(T56))
    print "Calculated:"
    print G.edge[6][4]['tfm'], '\n'
    
    # Let's try random angles:

    theta = -15 + np.random.random()*30
    theta2 = utils.rad_angle(theta)
    R1 = np.eye(4)
    R1[0:3,0:3] = rotation_matrix(zaxis,theta2)
    print "Theta: ",theta
    print "T71"
    print "Original:"
    print T7cor.dot(R1).dot(Tcor1)
    print "Calculated:"
    print G_opt.edge["master"]["l"]['tfm_func'](7,1,theta), '\n'
    print "norm diff:", nlg.norm(G_opt.edge["master"]["l"]['tfm_func'](7,1,theta)-T7cor.dot(R1).dot(Tcor1)), '\n'
    
    theta = -15 + np.random.random()*30
    theta2 = utils.rad_angle(theta)
    R1 = np.eye(4)
    R1[0:3,0:3] = rotation_matrix(zaxis,theta2)
    print "Theta: ",theta
    print "T71"
    print "Original:"
    print T7cor.dot(R1).dot(Tcor1)
    print "Calculated:"
    print G_opt.edge["master"]["l"]['tfm_func'](7,1,theta), '\n'
    print "norm diff:", nlg.norm(G_opt.edge["master"]["l"]['tfm_func'](7,1,theta)-T7cor.dot(R1).dot(Tcor1)), '\n'

    
    print "Difference between cors"
    T7cor_calib = G_opt.node["master"]["graph"].edge[7]["cor"]["tfm"]
    print nlg.inv(T7cor).dot(T7cor_calib)


if __name__=="__main__":
    test_gc3(10, True)
    