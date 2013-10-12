#!/usr/bin/ipython -i
import numpy as np, numpy.linalg as nlg

import scipy.optimize as sco
import networkx as nx, networkx.algorithms as nxa
import itertools
import time

import roslib; roslib.load_manifest("tf")
import rospy, tf

from hd_utils import utils, conversions
from hd_utils.yes_or_no import yes_or_no
from hd_utils.colorize import *

import get_marker_transforms as gmt 

np.set_printoptions(precision=5, suppress=True)

VERBOSE = 1

def update_graph_from_observations(G, tfms):
    """
    Updates transform graph @G based on the observations @tfms, which is a dict from 
    marker ids (or names) to transforms all in the same coordinate frame. 
    """
    
    ids = tfms.keys()
    
    if len(ids) == 1:
        G.add_node(ids[0])
        return
    
    ids.sort()
    
    for i,j in itertools.combinations(ids, 2):
        if not G.has_edge(i,j):
            G.add_edge(i,j)
            G.edge[i][j]['transform_list'] = []
            G.edge[i][j]['n'] = 0
        
        Tij = nlg.inv(tfms[i]).dot(tfms[j])
        #print "From ", i," to ", j,":\n",Tij
        G.edge[i][j]['transform_list'].append(Tij)
        G.edge[i][j]['n'] += 1


def update_groups_from_observations(masterGraph, tfms, pot_reading):
    """
    Updates graphs based on information.
    """
    ##
    # Should I do this?
    ##
    pot_reading = np.round(pot_reading)
    group_tfms = {}
    for group in masterGraph.nodes_iter():
        group_tfms[group] = {}
        for marker in masterGraph.node[group]["markers"]:
            if tfms.get(marker) is not None:
                group_tfms[group][marker] = tfms[marker]
        
        update_graph_from_observations(masterGraph.node[group]["graph"], group_tfms[group])
    
    for g1, g2 in itertools.combinations(masterGraph.nodes(),2):
        if masterGraph.node[g2].get("master_marker") is not None:
            g1, g2 = g2, g1

        tfms1 = group_tfms[g1]
        tfms2 = group_tfms[g2]
        if not tfms1 or not tfms2:
            continue
        
        for m1 in tfms1:
            for m2 in tfms2:
                if not masterGraph.has_edge(g1,g2):
                    masterGraph.add_edge(g1,g2)
                    masterGraph.edge[g1][g2]["transform_list"] = {}
                    masterGraph.edge[g1][g2]['n'] = 0
                if masterGraph.edge[g1][g2]["transform_list"].get(pot_reading) is None:
                    masterGraph.edge[g1][g2]["transform_list"][pot_reading] = []
                masterGraph.edge[g1][g2]["transform_list"][pot_reading].append({"from":m1,
                                                          "to":m2,
                                                          "tfm":nlg.inv(tfms1[m1]).dot(tfms2[m2])})
        masterGraph.edge[g1][g2]['n'] += 1


def is_ready (masterGraph, min_obs=5):
    """
    @num_markers is the total number of markers/sensors on the rigid object.
    @min_obs is the minimum number of observations required for each relative transform, once one is seen. 
    
    Returns True when the graph has enough data to begin calibration, False otherwise.
    """
    for group in masterGraph.nodes_iter():
        G = masterGraph.node[group]["graph"]
        if not G: 
            if VERBOSE: print group, "graph is null"
            return False
        if nx.is_connected(G) and G.number_of_nodes() == len(masterGraph.node[group]["markers"]):
            for i,j in G.edges_iter():
                if G.edge[i][j]['n'] < min_obs:
                    if VERBOSE:
                        print "edge", i, j, "of", group, "needs", min_obs - G.edge[i][j]['n'], "observations"
                    return False
        else:
            if VERBOSE:
                print group, "is not connected or does not have enough nodes"
            return false        
        
                
    for i,j in masterGraph.edges_iter():
        if masterGraph.edge[i][j]['n'] < min_obs:
            if VERBOSE:
                print "edge", i, j, "of masterGraph needs ", min_obs - masterGraph.edge[i][j]['n'], "observations"
            return False
        preadings = len(masterGraph.edge[i][j]["transform_list"])
        if preadings < min_obs:
            if VERBOSE:
                print "edge", i, j, "of masterGraph needs ", min_obs - preadings, "pot readings"
            return False
    return True

def optimize_transforms (G):
    """
    Optimize for transforms in G. Assumes G is_ready.
    Returns a clique with relative transforms between all objects.
    """

    if G.number_of_edges == 0:
        return G.to_directed()

    # Index maps from nodes and edges in the optimizer variable X.
    # Also calculate the reverse map if needed.
    node_map, edge_map = {}, {}
    rev_map = {}
    idx = 0

    for obj in G.nodes_iter():
        node_map[obj] = idx
        rev_map[idx] = obj
        idx += 1
    for i,j in G.edges_iter():
        if i < j:
            edge_map[i,j] = idx
            rev_map[idx] = i,j
        else:
            edge_map[j,i] = idx
            rev_map[idx] = j,i
        idx += 1
        
    # Some predefined variables
    I3 = np.eye(3)
        
    def get_mat_from_x(X, i, j):
        """
        X is a vertical stack of variables for transformation matrices (12 variables each).         
        Returns 3x4 matrix Ti or Tij by looking into the index maps.
        """

        if j is None:
            offset = node_map[i]*12
        else:
            if i < j:
                offset = edge_map[i,j]*12
            else:
                offset = edge_map[j,i]*12
        
        Xij = X[offset:offset+12]
        return Xij.reshape([3,4], order='F')


    def f_objective (X):
        """
        Objective function to make transforms close to average transforms.
        Sum of the norms of matrix differences between each Tij and 
        """        
        obj = 0
        for i,j in edge_map:
            obj += nlg.norm(get_mat_from_x(X, i, j) - G[i][j]['avg_tfm'][0:3,:])
            
        return obj
    
    def f_constraints (X):
        """
        Constraint function to force matrices to be valid transforms (R.T*R = I_3).
        It also forces Tj = Ti*Tij.
        Also, T0 = identity transform.
        """
        con = []
        Tis, Tijs = {},{}
        for node in node_map:
            Tis[node] = get_mat_from_x(X, node, None)
        for i,j in edge_map:
            Tijs[i,j] = get_mat_from_x(X, i, j)
        
        # T0 --> identity transform.
        con.append(nlg.norm(Tis[node_map.keys()[0]] - np.eye(4)[0:3,:]))
        
        # Constrain Ris to be valid rotations
        for node in node_map.keys()[1:]:
            Ri = Tis[node][0:3,0:3]
            con.append(nlg.norm(Ri.T.dot(Ri) - I3))
        
        # Tj = Ti*Tij
        # Rij.T*Rij = I
        for i,j in edge_map:
            Ti = np.r_[Tis[i],np.array([[0,0,0,1]])]
            Tj = np.r_[Tis[j],np.array([[0,0,0,1]])]
            Tij = np.r_[Tijs[i,j],np.array([[0,0,0,1]])]
            Rij = Tij[0:3,0:3]
              
            con.append(nlg.norm(Tj - Ti.dot(Tij)))
            con.append(nlg.norm(Rij.T.dot(Rij) - I3))
            
        return np.asarray(con)
    
    ### Setting initial values by walking through maximal spanning tree.
    G_init = G.copy()
    for i,j in G_init.edges_iter():
        G_init[i][j]['weight'] = -1*G_init[i][j]['n']
    G_init_tree = nxa.minimum_spanning_tree(G_init)
    
    x_init = np.zeros(12*(G.number_of_nodes() + G.number_of_edges()))
    
    node0 = G_init_tree.nodes()[0]
    offset0 = node_map[node0]
    x_init[offset0:offset0+9] = I3.reshape(9)
    
    fringe = [node0]
    seen = []
    while len(fringe)>0:
        node = fringe.pop(0)
        if node in seen: continue
        seen.append(node)
        for n_node in G_init_tree.neighbors(node):
            if n_node in seen: continue
            fringe.append(n_node)
            offset = node_map[n_node]*12
            tfm = G.edge[node][n_node]['avg_tfm']
            if node > n_node:
                tfm = nlg.inv(tfm)

            x_init[offset:offset+12] = tfm[0:3,:].reshape(12, order='F')
            
    for i,j in edge_map:
        offset = edge_map[i,j]*12
        x_init[offset:offset+12] = G.edge[i][j]['avg_tfm'][0:3,:].reshape(12,order='F')
    ### ###
    
    print "Initial x: ", x_init
    (X, fx, _, _, _) = sco.fmin_slsqp(func=f_objective, x0=x_init, f_eqcons=f_constraints, iter=100, full_output=1)
    
    # Create output optimal graph and copy edge transforms
    G_opt = G.to_directed()
    for i,j in edge_map:
        Tij = np.r_[get_mat_from_x(X, i, j),np.array([[0,0,0,1]])]
        Tji = nlg.inv(Tij)
        
        G_opt.edge[i][j] = {'tfm':Tij}
        G_opt.edge[j][i] = {'tfm':Tji}
        
    # Add all edges to make clique
    # Follow shortest path to get transform for edge
    for i,j in itertools.combinations(sorted(G_opt.nodes()), 2):
        if not G_opt.has_edge(i,j):
            ij_path = nxa.shortest_path(G_opt, i, j)
            Tij = G_opt.edge[ij_path[0]][ij_path[1]]['tfm']
            for p in xrange(2,len(ij_path)):
                Tij = Tij.dot(G_opt.edge[ij_path[p-1]][ij_path[p]]['tfm'])
            Tji = nlg.inv(Tij)
            
            G_opt.add_edge(i,j)
            G_opt.add_edge(j,i)
            G_opt.edge[i][j] = {'tfm':Tij}
            G_opt.edge[j][i] = {'tfm':Tji}
    
    for i in G_opt.nodes_iter():
        G_opt.add_edge(i,i)
        G_opt[i][i]['tfm'] = np.eye(4)
    
    n = G_opt.number_of_nodes()
    try:
        assert G_opt.number_of_edges() == n**2
    except AssertionError:
        print "Not all edges found...? Fix this"

    return G_opt

class tfmClass():
    def __init__(self, _group, mG_opt, master, masterG):
        self.group = _group
        self.childN = mG_opt.node[self.group]
        self.gpTfm = mG_opt.edge[master][self.group]['tfm']
        self.masterG = masterG
    
    def get_tfm(self, master_node, child_node, angle):
        # angle in degrees
        mtfm = self.masterG.edge[master_node]["cor"]['tfm']
        angle = utils.rad_angle(angle)
        rot = np.eye(4)
        rot[0:3,0:3] = utils.rotation_matrix(np.array([0,0,1]), angle*self.childN["angle_scale"])
        ctfm = self.childN["graph"].edge[self.childN["primary"]][child_node]['tfm']

        return mtfm.dot(rot).dot(self.gpTfm).dot(ctfm)

    # These two functions to make things pickleable
    def __getstate__(self):
        return {'group': self.group,
                'childN': self.childN,
                'gpTfm': self.gpTfm,
                'masterG': self.masterG}
        
    def __setstate__(self, state):
        self.group = state['group']
        self.childN = state['childN']
        self.gpTfm = state['gpTfm']
        self.masterG = state['masterG']


class tfmClassInv():
    def __init__(self, _group, mG_opt, master, masterG):
        self.group = _group
        self.childN = mG_opt.node[self.group]
        self.gpTfm = mG_opt.edge[master][self.group]['tfm']
        self.masterG = masterG
    
    def get_tfm(self, child_node, master_node, angle):
        # angle in degrees
        mtfm = self.masterG.edge[master_node]["cor"]['tfm']
        angle = utils.rad_angle(angle)
        rot = np.eye(4)
        rot[0:3,0:3] = utils.rotation_matrix(np.array([0,0,1]), angle*self.childN["angle_scale"])
        ctfm = self.childN["graph"].edge[self.childN["primary"]][child_node]['tfm']

        inv_tfm = mtfm.dot(rot).dot(self.gpTfm).dot(ctfm)
        
        return nlg.inv(inv_tfm)

    # These two functions to make things pickleable
    def __getstate__(self):
        return {'group': self.group,
                'childN': self.childN,
                'gpTfm': self.gpTfm,
                'masterG': self.masterG}
        
    def __setstate__(self, state):
        self.group = state['group']
        self.childN = state['childN']
        self.gpTfm = state['gpTfm']
        self.masterG = state['masterG']



def optimize_master_transforms (mG, init=None):
    """
    Optimize transforms over the masterGraph (which is a metagraph with nodes as rigid body graphs).
    
    """

    idx = 1
    node_map = {}
    master = None
    for node in mG.nodes_iter():
        if mG.node[node].get("master_marker") is not None:
            master = node 
            node_map[node] = 0
        else:
            node_map[node] = idx
            idx += 1
    
    # Some predefined variables
    I3 = np.eye(3)

    
    def get_mat_from_node (X, node, offset = None):
        """
        Get the matrix from the objective vector depending on the node.
        If offset is specified, return the matrix at the position directly.
        """
        if offset is not None:
            return X[offset:offset+12].reshape([3,4], order='F')

        offset = node_map[node]*12
        Xij = X[offset:offset+12]
        return Xij.reshape([3,4], order='F')
    
    def f_objective (X):
        """
        Objective function to make transforms close to average transforms.
        Sum of the norms of matrix differences between each Tij and 
        """
        obj = 0
        zaxis = np.array([0,0,1])
        
        for g1,g2 in mG.edges_iter():
            for angle in mG.edge[g1][g2]['avg_tfm']:
                t1 = np.r_[get_mat_from_node(X,g1),np.array([[0,0,0,1]])]
                t2 = np.r_[get_mat_from_node(X,g2),np.array([[0,0,0,1]])]

                rad = utils.rad_angle(angle)

                tot_angle = rad*(-mG.node[g1]["angle_scale"]+mG.node[g2]["angle_scale"])
                rot = np.eye(4)
                rot[0:3,0:3] = utils.rotation_matrix(zaxis, tot_angle)
                
                tfm = t1.dot(rot).dot(nlg.inv(t2))

                obj += nlg.norm(tfm - mG.edge[g1][g2]['avg_tfm'][angle])
        
        return obj
    
    def f_constraints (X):
        """
        Constraint function to force matrices to be valid transforms (R.T*R = I_3).
        """
        con = []
        # Constrain Ris to be valid rotations
        
        for node in node_map:
            Ri = get_mat_from_node(X, node)[0:3,0:3]
            con.append(nlg.norm(Ri.T.dot(Ri) - I3))
            
        return np.asarray(con)    
    
    ## Intial value assumes each rigid body has some edge with master.
    ## Averaging out over angles.
    if init is not None:
        x_init = init
    else:
        x_init = np.zeros(12*mG.number_of_nodes())
        x_init[0:9] = I3.reshape(9)
        for node in mG.neighbors_iter(master):
            init_tfm = utils.avg_transform(mG.edge[master][node]['avg_tfm'].values())
            print init_tfm.dot(np.r_[np.c_[np.eye(3),np.array([0,0,0])],np.array([[0,0,0,0]])]).dot(init_tfm.T)
            offset = node_map[node]*12
            x_init[offset:offset+12] = init_tfm[0:3,:].reshape(12, order='F')
        ##

    print "Initial x: ", x_init
    print "Initial objective: ", f_objective(x_init)
    
    (X, fx, _, _, _) = sco.fmin_slsqp(func=f_objective, x0=x_init, f_eqcons=f_constraints, iter=200, full_output=1, iprint=2)
    
    mG_opt = nx.DiGraph()
    ## modify master graph
    node0 = mG.node[master]["primary"]
    
    masterG = mG.node[master]["graph"]
    node_iter = masterG.neighbors(node0) 
    masterG.add_node("cor")
    # change master primary
    mG.node[master]["primary"] = "cor"
    masterG.add_edge(node0,"cor")
    masterG.add_edge("cor",node0)
    masterG.add_edge("cor","cor")
    
    Tsc = np.r_[get_mat_from_node(X, master), np.array([[0,0,0,1]])] 
    masterG.edge[node0]["cor"]['tfm'] = Tsc
    masterG.edge["cor"][node0]['tfm'] = nlg.inv(Tsc)
    masterG.edge["cor"]["cor"]['tfm'] = np.eye(4)
    
    for node in node_iter:
        masterG.add_edge(node,"cor")
        masterG.add_edge("cor",node)
        
        tfm = masterG.edge[node][node0]['tfm'].dot(Tsc)
        masterG.edge[node]["cor"]['tfm'] = tfm
        masterG.edge["cor"][node]['tfm'] = nlg.inv(tfm)
        
    mG_opt.add_node(master)
    mG_opt.node[master]["graph"] = masterG
    mG_opt.node[master]["angle_scale"] = 0
    mG_opt.node[master]["primary"] = "cor"
    mG_opt.node[master]["master_marker"] = mG.node[master]["master_marker"]
    mG_opt.node[master]["markers"] = mG.node[master]["markers"]
    mG_opt.node[master]["hydras"] = mG.node[master]["hydras"]
    mG_opt.node[master]["ar_markers"] = mG.node[master]["ar_markers"]
    ## add edges to the rest
        
    
    for group in mG.nodes_iter():
        if group == master: continue
        mG_opt.add_edge(master, group)
        mG_opt.add_edge(group, master)
        mG_opt.node[group]["graph"] = mG.node[group]["graph"]
        mG_opt.node[group]["angle_scale"] = mG.node[group]["angle_scale"]
        mG_opt.node[group]["primary"] = mG.node[group]["primary"]
        mG_opt.node[group]["hydras"] = mG.node[group]["hydras"]
        mG_opt.node[group]["ar_markers"] = mG.node[group]["ar_markers"]
        
        tfm = np.r_[get_mat_from_node(X, group), np.array([[0,0,0,1]])]
        mG_opt.edge[group][master]['tfm'] = tfm
        mG_opt.edge[master][group]['tfm'] = nlg.inv(tfm)

        tfmFuncs = tfmClass(group,  mG_opt, master, masterG)
        tfmFuncsInv = tfmClassInv(group,  mG_opt, master, masterG)
        mG_opt.edge[master][group]['tfm_func'] = tfmFuncs
        mG_opt.edge[group][master]['tfm_func'] = tfmFuncsInv
        
    
    return mG_opt
                
        

    
def compute_relative_transforms (masterGraph, init=None):
    """
    Takes in a transform graph @G such that it has enough data to begin calibration (is_ready(G) returns true).
    Optimizes and computes final relative transforms between all nodes (markers).
    Returns a graph final_G with the all edges (clique) and final transforms stored in edges.
    Make sure the graph is ready before this by calling is_ready(graph,min_obs).
    """

    new_mG = nx.DiGraph()
    graph_map = {} 
    for group in masterGraph.nodes_iter():
        G = masterGraph.node[group]["graph"]
        for i,j in G.edges_iter():
            G[i][j]['avg_tfm'] = utils.avg_transform(G[i][j]['transform_list'])
        # Optimize rigid body transforms.
        graph_map[G] = optimize_transforms(G)
        new_mG.add_node(group)
        new_mG.node[group]["graph"] = graph_map[G]
        if masterGraph.node[group].get("master_marker"):
            new_mG.node[group]["master_marker"] = masterGraph.node[group].get("master_marker")
        new_mG.node[group]["angle_scale"] = masterGraph.node[group]["angle_scale"]
        new_mG.node[group]["markers"] = masterGraph.node[group]["markers"]
        
        new_mG.node[group]["hydras"] = masterGraph.node[group]["hydras"]
        new_mG.node[group]["ar_markers"] = masterGraph.node[group]["ar_markers"]

        masterGraph.node[group]["primary"] = masterGraph.node[group]["graph"].nodes()[0]
        new_mG.node[group]["primary"] = masterGraph.node[group]["primary"]

    for g1,g2 in masterGraph.edges_iter():
        mg1  = graph_map[masterGraph.node[g1]["graph"]]
        mg2  = graph_map[masterGraph.node[g2]["graph"]]
        new_mG.add_edge(g1, g2)

        node1 = new_mG.node[g1].get("primary")
        node2 = new_mG.node[g2].get("primary")
        
        new_mG[g1][g2]['avg_tfm'] = {}
        for angle, tfm_data in masterGraph[g1][g2]['transform_list'].items():
            transforms = []
            # Converting all transforms to standard transform between two fixed nodes.
            for tfm in tfm_data:
                transforms.append(mg1.edge[node1][tfm['from']]['tfm'].dot(tfm['tfm'])\
                                  .dot(mg2.edge[tfm['to']][node2]['tfm']))
            new_mG[g1][g2]['avg_tfm'][angle] = utils.avg_transform(transforms)

    # Optimize over angles to get master graph with transforms.
    mG_opt = optimize_master_transforms(new_mG, init)
    
    return mG_opt

class Gripper:
    
    # Change if need be
    parent_frame = 'camera1_rgb_optical_frame'
    
    cameras = None
    
    transform_graph = None
    tt_calculated = False
    lr = None
    mmarkers = []
    lmarkers = []
    rmarkers = []
    allmarkers = []
    
    ar_markers = []
    hydra_markers= []
    
    def __init__ (self, lr, transform_graph, cameras=None):
        self.transform_graph = transform_graph
        self.cameras = cameras
        if self.cameras is not None:
            self.parent_frame = self.cameras.parent_frame

        for group in self.transform_graph.nodes_iter():
            self.ar_markers.extend(self.transform_graph.node[group]['ar_markers'])
            self.hydra_markers.extend(self.transform_graph.node[group]['hydras'])

        self.mmarkers = self.transform_graph.node['master']['graph'].nodes()        
        self.lmarkers = self.transform_graph.node['l']['graph'].nodes()
        self.rmarkers = self.transform_graph.node['r']['graph'].nodes()
        self.allmarkers = self.mmarkers + self.rmarkers + self.lmarkers
        self.lr = lr
        
    def set_cameras (self, cameras):
        self.cameras = cameras
        
    
    def calculate_tool_tip_transform (self, m1, m2):
        """
        Assuming that the gripper has "master", "l" and "r"
        m1 and m2 are markers on the tool tip. 
        """
        if self.tt_calculated:
            yellowpring("Tool tip already calculated.")
            return
        if m1 in self.rmarkers:
            assert m2 in self.lmarkers
            m1, m2 = m2, m1
        else:
            assert m1 in self.lmarkers
            assert m2 in self.rmarkers

        # Assume that the origin of the tool tip transform is the avg
        # point of fingers when they're at an angle of 0
        ltfm = self.transform_graph.edge['master']['l']['tfm_func'].get_tfm('cor', m1,0)
        rtfm = self.transform_graph.edge['master']['r']['tfm_func'].get_tfm('cor', m2,0)
        
        lorg = ltfm[0:3,3]
        rorg = ltfm[0:3,3]
        tt_org = (lorg+rorg)/2.0

        master = self.transform_graph.node['master']['graph']
        
        # Old "correct" code.
        # z axis is the same as cor
        #tt_z = np.array([0,0,1])
        # x axis (pointing axis) is the projection of vector from cor to tt_org on the xy plane.
        # Since we're in the 'cor' frame, it's just the projection of direction vector tt_org on xy plane.
        #tt_org_vec = tt_org/nlg.norm(tt_org)
        #tt_x = tt_org_vec - tt_org_vec.dot(tt_z)*tt_z
        #tt_x = tt_x/nlg.norm(tt_x)
        #tt_y = np.cross(tt_z, tt_x)
        #tt_tfm =  np.r_[np.c_[tt_x, tt_y, tt_z, tt_org], np.array([[0,0,0,1]])]
        

        # New hacky code:
        master_marker = self.transform_graph.node['master']['master_marker']
        master_tfm = master.edge['cor'][master_marker]['tfm']
        tt_tfm =  np.r_[np.c_[master_tfm[0:3,0:3], tt_org], np.array([[0,0,0,1]])]
        
        
        master.add_node('tool_tip')
        master.add_edge('cor', 'tool_tip')
        master.add_edge('tool_tip', 'cor')
        
        master.edge['cor']['tool_tip']['tfm'] = tt_tfm
        master.edge['tool_tip']['cor']['tfm'] = nlg.inv(tt_tfm)
        
        for node in master.nodes_iter():
            if node == 'cor' or node == 'tool_tip': continue
            master.add_edge(node, 'tool_tip')
            master.add_edge('tool_tip', node)
            
            tfm = master.edge[node]['cor']['tfm'].dot(tt_tfm)
        
            master.edge[node]['tool_tip']['tfm'] = tfm
            master.edge['tool_tip'][node]['tfm'] = nlg.inv(tfm)
        
        self.mmarkers.append('tool_tip')
        self.allmarkers.append('tool_tip')
        self.tt_calculated = True

        
    def get_tool_tip_transform(self, marker_tfms, theta):
        """
        Get tool tip transfrom from dict of relevant markers to transforms
        Also provide the angle of gripper.
        Don't need it if only master transform is visible.
        m has to be on one of the 'master,'l' or 'r' groups
        """
        if self.tt_calculated is False:
            redprint ("Tool tip transform not calibrated.")
            return

        masterg = self.transform_graph.node['master']['graph']

        avg_tfms = []
        
        for m in marker_tfms:
            if m in self.mmarkers:
                tt_tfm = masterg.edge[m]['tool_tip']['tfm']
            elif m in self.lmarkers:
                tt_tfm = self.transform_graph.edge['l']['master']['tfm_func'].get_tfm(m,'tool_tip', theta)
            elif m in self.rmarkers:
                tt_tfm = self.transform_graph.edge['r']['master']['tfm_func'].get_tfm(m,'tool_tip', theta)
            else:
                redprint('Marker %s not on gripper.'%m)
                continue
            avg_tfms.append(marker_tfms[m].dot(tt_tfm))
            
        if len(avg_tfms) == 0:
            redprint('No markers on gripper found.')
            return

        return utils.avg_transform(avg_tfms)
    
    def get_rel_transform(self, m1, m2, theta):
        """
        Return relative transform between any two markers on gripper.
        """
        masterg = self.transform_graph.node['master']['graph']

        if m1 in self.mmarkers:
            tfm1 = masterg.edge[m1]['cor']['tfm']
        elif m1 in self.lmarkers:
            tfm1 = self.transform_graph.edge['l']['master']['tfm_func'].get_tfm(m1,'cor', theta)
        elif m1 in self.rmarkers:
            tfm1 = self.transform_graph.edge['r']['master']['tfm_func'].get_tfm(m1,'cor', theta)
        else:
            redprint('Marker %s not on gripper.'%m1)
            return

        if m2 in self.mmarkers:
            tfm2 = masterg.edge['cor'][m2]['tfm']
        elif m2 in self.lmarkers:
            tfm2 = self.transform_graph.edge['master']['l']['tfm_func'].get_tfm('cor', m2, theta)
        elif m2 in self.rmarkers:
            tfm2 = self.transform_graph.edge['master']['r']['tfm_func'].get_tfm('cor', m2, theta)
        else:
            redprint('Marker %s not on gripper.'%m2)
            return
        
        return tfm1.dot(tfm2)
    
    def get_all_transforms(self, parent_frame):
        """
        From marker transforms given in parent_frame, get all transforms
        of all markers on gripper.
        """
        
        ar_tfms = self.cameras.get_ar_markers()
        hyd_tfms = gmt.get_hydra_transforms(self.parent_frame, None)
        theta = gmt.get_pot_angle()
        
        marker_tfms= ar_tfms
        marker_tfms.update(hyd_tfms)
        
        ret_tfms = []
        
        cor_avg_tfms = []
        cor_hyd_avg = []
        cor_ar_avg = []
        ar_found = False
        hyd_found = False
        for m,tfm in marker_tfms.items():
            if m in self.ar_markers:
                c_tfm = tfm.dot(self.get_rel_transform(m,'cor', theta)) 
                cor_ar_avg.append(c_tfm)
                cor_avg_tfms.append(c_tfm)
                ar_found = True
            elif m in self.hydra_markers:
                c_tfm = tfm.dot(self.get_rel_transform(m,'cor', theta)) 
                cor_hyd_avg.append(c_tfm)
                cor_avg_tfms.append(c_tfm)
                hyd_found = True
        
        if len(cor_avg_tfms) == 0: return ret_tfms
        
        cor_tfm = utils.avg_transform(cor_avg_tfms)
        cor_h_tfm = utils.avg_transform(cor_hyd_avg)
        cor_a_tfm = utils.avg_transform(cor_ar_avg)
        
        ret_tfms.append({'parent':parent_frame, 
                         'child':'%sgripper_%s'%(self.lr, 'cor'),
                         'tfm':cor_tfm})

        for m in self.allmarkers:
            if m != 'cor' and m != 'tool_tip':
                tfm = self.get_rel_transform('cor', m, theta)
                ret_tfms.append({'parent':parent_frame, 
                                 'child':'%sgripper_%s'%(self.lr, m),
                                 'tfm':cor_tfm.dot(tfm)})

        if self.tt_calculated:
            tfm = self.get_rel_transform('cor', 'tool_tip', theta)
            ret_tfms.append({'parent':parent_frame, 
                             'child':'%sgripper_tooltip'%self.lr,
                             'tfm':cor_tfm.dot(tfm)})
            if hyd_found:
                ret_tfms.append({'parent':parent_frame, 
                                 'child':'%sgripper_tooltip_hydra'%self.lr,
                                 'tfm':cor_h_tfm.dot(tfm)})
            if ar_found:
                ret_tfms.append({'parent':parent_frame, 
                                 'child':'%sgripper_tooltip_ar'%self.lr,
                                 'tfm':cor_a_tfm.dot(tfm)})

        return ret_tfms
    
    def get_specific_tool_tip_tfm (self, m_type='AR'):
        """
        Get the estimate of the tool tip from either ar_markers or hydras.
        """
        if m_type not in ['AR', 'hydra']:
            redprint('Not sure what estimate %s gives.'%m_type)
            return
        
        if m_type == 'AR':
            if len(self.ar_markers) == 0:
                redprint('No AR markers to give you an estimate.')
                return
            marker_tfms = self.cameras.get_ar_markers(markers=self.ar_markers)
        else:
            if len(self.hydra_markers) == 0:
                redprint('No hydras to give you an estimate.')
                return
            marker_tfms = gmt.get_hydra_transforms(self.parent_frame, self.hydra_markers)
        
        theta = gmt.get_pot_angle()
        return self.get_tool_tip_transform(marker_tfms, theta)
        
    
    def get_markers_transform (self, markers, marker_tfms, theta):
        """
        Takes in marker_tfms found, angles and markers for
        which transforms are required.
        Returns a dict of marker to transforms for the markers
        requested.
        
        """
        
        rtn_tfms = {}
        cor_avg_tfms = []
        for m,tfm in marker_tfms.items():
            if m in self.allmarkers:
                cor_avg_tfms.append(tfm.dot(self.get_rel_transform(m,'cor', theta)))
        
        cor_tfm = utils.avg_transform(cor_avg_tfms)

        for marker in markers:
            if marker in marker_tfms:
                rtn_tfms[marker] = marker_tfms[marker]
                continue
            
            tfm = self.get_rel_transform('cor', marker, gmt.get_pot_angle())
            if tfm is not None:
                rtn_tfms[marker] = tfm
        
        return rtn_tfms

        
    
    def get_obs_new_marker(self, marker, n_tfm = 10, n_avg=10):
        """
        Store a bunch of readings seen for new marker being added.
        Returns a list of dicts each of which has marker transforms found
        and potentiometer angles.
        """
        
        all_obs = []
        
        i = 0
        thresh = n_avg*2
        sleeper = rospy.Rate(30)
        
        while i < n_tfm:
            raw_input(colorize("Getting transform %i out of %i. Hit return when ready."%(i,n_tfm), 'yellow', True))
            
            j = 0
            j_th = 0
            pot_angle = 0
            found = True
            avg_tfms = []
            while j < n_avg:
                blueprint("Averaging transform %i out of %i."%(j,n_avg))
                ar_tfms = self.cameras.get_ar_markers();
                hyd_tfms = gmt.get_hydra_transforms(parent_frame=self.parent_frame, hydras=None);
                curr_angle = gmt.get_pot_angle()
                
                if not ar_tfms and not hyd_tfms:
                    yellowprint("Could not find any transform.")
                    j_th += 1
                    if j_th < thresh:
                        continue
                    else:
                        found = False
                        break
                                
                tfms = ar_tfms
                pot_angle += curr_angle
                tfms.update(hyd_tfms)

                avg_tfms.append(tfms)
                j += 1
                sleeper.sleep()
            
            if found is False:
                yellowprint("Something went wrong; try again.")
                continue
            
            tfms_found = {}
            for tfms in avg_tfms:
                for m in tfms:
                    if m not in tfms_found:
                        tfms_found[m] = []
                    tfms_found[m].append(tfms[m])

            if marker not in tfms_found:
                yellowprint("Could not find marker to be added; try again.")
                continue
            
            for m in tfms_found:
                tfms_found[m] = utils.avg_transform(tfms_found[m])
            pot_angle = pot_angle/n_avg
            
            all_obs.append({'tfms':tfms_found, 'pot_angle':pot_angle})
            i += 1
            
        return all_obs
            

    
    def add_marker (self, marker, group, m_type='AR', n_tfm=10, n_avg=10):
        """
        Add marker to the gripper, after calibration.
        Observation + calculation + adding new marker all done here.
        Maybe should break this up into smaller functions.
        """
        #Checks to see if data + situation is valid to add marker
        if marker in self.allmarkers:
            reenprint("Marker already added.")
            return

        if self.cameras is None:
            redprint("Sorry, have no cameras to check.")
            return
        
        if group not in ['master','l','r']:
            redprint('Invalid group %s'%group)
            return
        
        if m_type not in ['AR', 'hydra']:
            redprint('Invalid marker type %s'%m_type)
            return

        # Get observations 
        all_obs = self.get_obs_new_marker(marker, n_tfm, n_avg)
        print all_obs
        
        # Compute average relative transform between correct primary node and marker
        primary_node = self.transform_graph.node[group]['primary']
        avg_rel_tfms = []
        
        for obs in all_obs:
            tfms = obs['tfms']
            angle = obs['tfms']
            
            marker_tfm = tfms[marker]
            primary_tfm = self.get_markers_transform([primary_node], tfms, angle)[primary_node]
            
            avg_rel_tfms.append(nlg.inv(primary_tfm).dot(marker_tfm))
            
        rel_tfm = utils.avg_transform(avg_rel_tfms)
        
        # Add markers to relevant lists
        self.transform_graph.node[group]['markers'].append(marker)
        self.allmarkers.append(marker)
        
        if m_type=='AR':
            self.transform_graph.node[group]['ar_markers'].append(marker)
            self.ar_markers.append(marker)
        else:
            self.transform_graph.node[group]['hydras'].append(marker)
            self.hydra_markers.append(marker)
            
        if group=='master':
            self.mmarkers.append(marker)
        elif group=='l':
            self.lmarkers.append(marker)
        else:
            self.rmarkers.append(marker)

        # Update graphs from transform found
        graph = self.transform_graph.node[group]['graph']
        graph.add_node(marker)
        
        for node in graph.nodes_iter():
            graph.add_edge(node, marker)
            graph.add_edge(marker, node)
            
            tfm = graph.edge[node][primary_node]['tfm'].dot(rel_tfm)
            
            graph.edge[node][marker]['tfm'] = tfm
            graph.edge[marker][node]['tfm'] = nlg.inv(tfm)
        
        greenprint("Successfully added your marker to the gripper!")
            
        


class GripperCalibrator:
    
    calib_info = None
    
    lr = None
    
    masterGraph = None
    transform_graph = None
    ar_markers = []
    hydras = []
    iterations = 0
    
    parent_frame = None
    
    cameras = None
    calibrated = False
    
    tt_calculated = False
    
    gripper = None
    
    def __init__(self, cameras, lr = 'l', calib_info=None, parent_frame = 'camera1_rgb_optical_frame'):
        self.cameras = cameras
        self.parent_frame = parent_frame
        self.calib_info = calib_info
        self.lr = lr
    
    def update_calib_info (self, calib_info):
        self.reset_calibration()
        self.calib_info = calib_info
        
    def initialize_calibration (self, fake_data=False):
        if not fake_data:
            assert self.cameras.calibrated
            # assert hydras are calibrated

        self.masterGraph = nx.DiGraph()
        
        for group in self.calib_info:
            self.masterGraph.add_node(group)
            self.masterGraph.node[group]["graph"] = nx.Graph()
            self.masterGraph.node[group]["angle_scale"] = self.calib_info[group]['angle_scale']
            
            if self.calib_info[group].get("ar_markers") is None:
                self.calib_info[group]["ar_markers"] = []
            if self.calib_info[group].get("hydras") is None:
                self.calib_info[group]["hydras"] = []
            
            if self.calib_info[group].get("master_marker") is not None:
                self.masterGraph.node[group]["master_marker"] = self.calib_info[group].get("master_marker")
                self.masterGraph.node[group]['angle_scale'] = 0

            self.masterGraph.node[group]['hydras'] = self.calib_info[group]["hydras"]
            self.masterGraph.node[group]['ar_markers'] = self.calib_info[group]["ar_markers"]     
            self.masterGraph.node[group]["markers"] = self.calib_info[group]["ar_markers"] + self.calib_info[group]["hydras"]
            
            self.ar_markers.extend(self.calib_info[group]["ar_markers"])
            self.hydras.extend(self.calib_info[group]["hydras"])


    
    def process_observation (self, n_avg=5):
        self.iterations += 1
        raw_input(colorize("Iteration %d: Press return when ready to capture transforms."%self.iterations, "red", True))
        
        sleeper = rospy.Rate(30)
        avg_tfms = {}
        j = 0
        thresh = n_avg*2
        pot_avg = 0.0
        while j < n_avg:
            blueprint('\tGetting averaging transform : %d of %d ...'%(j,n_avg-1))    

            tfms = {}
            # Fuck the frames bullshit
            ar_tfm = self.cameras.get_ar_markers(markers=self.ar_markers)
            hyd_tfm = gmt.get_hydra_transforms(parent_frame=self.parent_frame, hydras = self.hydras)
            pot_angle = gmt.get_pot_angle()
            
            if not ar_tfm or (not hyd_tfm and self.hydras):
                if not ar_tfm:
                    yellowprint('Could not find all required ar markers.')
                else:
                    yellowprint('Could not find all required hydra transforms.')
                thresh -= 1
                if thresh == 0: return False
                continue
            
            
            pot_avg += pot_angle
            
            j += 1
            tfms.update(ar_tfm)
            tfms.update(hyd_tfm)

            # The angle relevant for each finger is only half the angle.

                        
            for marker in tfms:
                if marker not in avg_tfms:
                    avg_tfms[marker] = []
                avg_tfms[marker].append(tfms[marker])
                
            sleeper.sleep()
            
        pot_avg /= n_avg
        
        greenprint("Average angle found: %f"%pot_avg)

        for marker in avg_tfms:
            avg_tfms[marker] = utils.avg_transform(avg_tfms[marker])

        update_groups_from_observations(self.masterGraph, avg_tfms, pot_angle)
        return True


    def finish_calibration (self):
        """
        Finishes calibration by performing the optimization.
        Make sure graph is ready before running this by checking is_ready(graph,min_obs)
        Takes several seconds.
        """
        self.transform_graph = compute_relative_transforms(self.masterGraph)
        return True


    def calibrate (self, min_obs=5, n_avg=5):
        self.initialize_calibration()

        while True:
            worked = self.process_observation(n_avg)
            if not worked:
                yellowprint("Something went wrong. Try again.")
                self.iterations -= 1
            if is_ready(self.masterGraph, min_obs):
                if yes_or_no("Enough data has been gathered. Proceed with transform optimization?"):
                    break

        assert is_ready (self.masterGraph, min_obs)
        self.calibrated = self.finish_calibration()


    def reset_calibration (self):
        self.calibrated = False
        self.calib_info = None

        self.gripper = None
        self.masterGraph = None
        self.transform_graph = None
        self.ar_markers = []
        self.hydras = []
        self.iterations = 0

    def get_transform_graph (self):
        
        if not self.calibrated:
            redprint("Gripper not calibrated.")
            return

        return self.transform_graph

    def get_gripper (self):
        if not self.calibrated:
            redprint("Gripper not calibrated.")
            return
        if self.gripper is None:
            self.gripper = Gripper(self.lr, self.transform_graph, self.cameras)
        
        return self.gripper