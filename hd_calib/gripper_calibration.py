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
from hd_utils.colorize import colorize

tf_listener = None

np.set_printoptions(precision=5, suppress=True)

def update_graph_from_observations(G, tfms):
    """
    Updates transform graph @G based on the observations @tfms, which is a dict from 
    marker ids (or names) to transforms all in the same coordinate frame. 
    """
    ids = tfms.keys()
    ids.sort()
    
    for i,j in itertools.combinations(ids, 2):
        if not G.has_edge(i,j):
            G.add_edge(i,j)
            G.edge[i][j]['transform_list'] = []
            G.edge[i][j]['n'] = 0
        
        Tij = nlg.inv(tfms[i]).dot(tfms[j])
        print "From ", i," to ", j,":\n",Tij
        G.edge[i][j]['transform_list'].append(Tij)
        G.edge[i][j]['n'] += 1


def update_groups_from_observations(masterGraph, group_info, tfms, pot_reading):
    """
    Updates graphs based on information.
    """
    ##
    # Should I do this?
    ##
    pot_reading = np.round(pot_reading)
    group_tfms = {}
    for group in group_info:
        group_tfms[group] = {}
        for marker in group_info[group]["markers"]:
            if tfms.get(marker) is not None:
                group_tfms[group][marker] = tfms[marker]
        
        update_graph_from_observations(group_info[group]["graph"], group_tfms[group])
    
    for g1, g2 in itertools.combinations(masterGraph.nodes(),2):
        if masterGraph.node[g2].get("master") is not None:
            g1, g2 = g2, g1

        tfms1 = group_tfms[g1]
        tfms2 = group_tfms[g2]
        if not tfms1 or tfms2:
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
                


def is_ready (masterGraph, group_info, min_obs=5):
    """
    @num_markers is the total number of markers/sensors on the rigid object.
    @min_obs is the minimum number of observations required for each relative transform, once one is seen. 
    
    Returns True when the graph has enough data to begin calibration, False otherwise.
    """
    for group in group_info:
        G = group_info[group]["graph"]
        if nx.is_connected() and G.number_of_nodes() == len(group_info[group]["markers"]):
            for i,j in G.edges_iter():
                if G.edge[i][j]['n'] < min_obs:
                    return False
                
    for i,j in masterGraph.edge_iter():
        if masterGraph.edge[i][j]['n'] < min_obs:
            return False
    return True

def optimize_transforms (G):
    """
    Optimize for transforms in G. Assumes G is_ready.
    Returns a clique with relative transforms between all objects.
    """

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
    
    n = G_opt.number_of_nodes()
    try:
        assert G_opt.number_of_edges() == n*(n-1)
    except AssertionError:
        print "Not all edges found...? Fix this"

    return G_opt
            
def optimize_master_transforms (mG):
    """
    Optimize transforms over the masterGraph (which is a metagraph with nodes as rigid body graphs).
    """

    idx = 1
    node_map = {}
    master = None
    for node in mG.nodes_iter():
        if mG.node[node].get("master") is not None:
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
                tot_angle = rad*(-mG.node[g1]["scale_factor"]+mG.node[g2]["scale_factor"])
                rot = utils.rotation_matrix(zaxis, tot_angle)
                
                tfm = t1.dot(rot).dot(nlg.inv(t2))
                
                obj += nlg.norm(tfm - mG[g1][g2]['avg_tfm'][angle])
            
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
    x_init = np.zeros(12*mG.number_of_nodes())
    x_init[0:9] = I3.reshape(9)
    for node in mG.neighbors_iter(master):
        init_tfm = utils.avg_transform(mG.edge[master][node]['avg_tfm'].values())
        offset = node_map[node]
        x_init[offset:offset+12] = init_tfm[0:3,:].reshape(12, order='F')
    ##
    
    print "Initial x: ", x_init
    (X, fx, _, _, _) = sco.fmin_slsqp(func=f_objective, x0=x_init, f_eqcons=f_constraints, iter=100, full_output=1)
    
    mG_opt = nx.DiGraph()
    ## modify master graph
    node0 = mG.node()[master].get("primary")
    node_iter = master.neighbors_iter(node0) 
    
    master.add_node("cor")
    # change master primary
    mG.node[master]["primary"] = "cor"
    master.add_edge(node0,"cor")
    master.add_edge("cor",node0)
    
    Tsc = np.r_[get_mat_from_node(X, master), np.array([0,0,0,1])] 
    master.edge[node0]["cor"]['tfm'] = Tsc
    master.edge["cor"][node0]['tfm'] = nlg.inv(Tsc)
    
    for node in node_iter:
        master.add_edge(node,"cor")
        master.add_edge("cor",node)
        
        tfm = master.edge[node][node0]['tfm'].dot(Tsc)
        master.edge[node]["cor"] = tfm
        master.edge["cor"][node] = nlg.inv(tfm)
        
    mG_opt.add_node(master)
    mG_opt.node[master]["scale_factor"] = 0
    mG_opt.node[master]["primary"] = "cor"
    ## add edges to the rest
    for G in mG.nodes_iter():
        if G == master: continue
        mG_opt.add_edge(master, G)
        mG_opt.add_edge(G, master)
        mG_opt.node[G]["scale_factor"] = mG.node[G]["scale_factor"]
        mG_opt.node[G]["primary"] = mG.node[G]["primary"]
        
        tfm = np.r_[get_mat_from_node(X, G), np.array([0,0,0,1])]
        mG_opt.edge[G][master]['tfm'] = tfm
        mG_opt.edge[master][G]['tfm'] = nlg.inv(tfm)
        
        def get_tfm(master_node, child_node, angle):
            mprimary_node = mG_opt.node[master]["primary"] 
            if mprimary_node == master_node:
                mtfm = np.eye(4)
            else:
                mtfm = master.edge[master_node][mprimary_node]['tfm']
            
            rot = utils.rotation_matrix(np.array([0,0,0,1]), angle*mG_opt[G]["scale_factor"])
            
            cprimary_node = mG_opt.node[G]["primary"] 
            if cprimary_node == child_node:
                ctfm = np.eye(4)
            else:
                ctfm = G.edge[child_node][cprimary_node]['tfm']
                
            return mtfm.dot(rot).dot(ctfm)
        
        def get_tfm_inv(child_node, master_node, angle):
            return nlg.inv(get_tfm(master_node,child_node,angle))
        
        mG_opt.edge[master][G]['tfm_func'] = get_tfm
        mG_opt.edge[G][master]['tfm_func'] = get_tfm_inv
        
    return mG_opt
                
        

    
def compute_relative_transforms (masterGraph, group_info, min_obs=5):
    """
    Takes in a transform graph @G such that it has enough data to begin calibration (is_ready(G) returns true).
    Optimizes and computes final relative transforms between all nodes (markers).
    Returns a graph final_G with the all edges (clique) and final transforms stored in edges.
    """

    assert (is_ready(masterGraph, group_info, min_obs=min_obs))

    new_mG = nx.DiGraph()
    graph_map = {} 
    for G in masterGraph.nodes():
        for i,j in G.edges_iter():
            G[i][j]['avg_tfm'] = utils.avg_transform(G[i][j]['transform_list'])
        # Optimize rigid body transforms.
        graph_map[G] = optimize_transforms(G)
        new_mG.add_node(graph_map[G])
        if masterGraph.node[G].get("master"):
            new_mG.node[graph_map[G]]["master"] = 1
        new_mG.node[graph_map[G]]["scale_factor"] = masterGraph.node[G]["scale_factor"] 
    
    for g in masterGraph.nodes_iter():
        masterGraph.node[g]["primary"] = g.nodes()[0]

    for g1,g2 in masterGraph.edges_iter():
        mg1  = graph_map[g1]
        mg2  = graph_map[g2]
        new_mG.add_edge(mg1, mg2)
    
        node1 = new_mG.node[mg1].get("primary")
        node2 = new_mG.node[mg2].get("primary")
        
        new_mG[mg1][mg2]['avg_tfm'] = {}
        for angle, tfm_data in masterGraph[g1][g2]['transform_list'].items():
            transforms = []
            # Converting all transforms to standard transform between two fixed nodes.
            for tfm in tfm_data:
                transforms.append(mg1.edge[node1][tfm['from']]['tfm'].dot(tfm['tfm']).dot(mg2.edge[tfm['to']][node2]['tfm']))
            new_mG[mg1][mg2]['avg_tfm'][angle] = utils.avg_transform(transforms)

    # Optimize over angles to get master graph with transforms.
    mG_opt = optimize_master_transforms(new_mG)
    
    return mG_opt


# These three functions assume calibration has taken place.
def get_ar_transforms(markers, parent_frame):
    """
    Takes in a list of @markers (AR marker ids) and a @parent_frame.
    Returns a dict of transforms with keys as found marker ids and values as transforms.
    """
    if markers is None: return {}
    ar_tfms = {}
    for marker in markers:
        try:
            trans, rot = tf_listener.lookupTransform(parent_frame, 'ar_marker_%d'%marker, rospy.Time(0))
            ar_tfms[marker] = conversions.trans_rot_to_hmat(trans, rot)
        except:
            pass
    return ar_tfms
    
def get_hydra_transforms(hydras, parent_frame):
    """
    Transform finder for hydras. Nothing for now.
    """
    return {}

def get_phasespace_transforms(ps_markers, parent_frame):
    """
    Transform finder for hydras. Nothing for now.
    """
    return {}

def get_potentiometer_angle():
    """
    Finds the angle of the potentiometer. Nothing yet.
    """
    return 0


def create_graph_from_observations(parent_frame, calib_info, min_obs=5, n_avg=5, freq=None):
    """
    Runs a loop till graph has enough data and user is happy with data.
    Or run with frequency specified until enough data is gathered.
    
    @parent_frame -- frame to get observations in.
    @num_markers -- total_number of markers.
    @calib_info -- dict with information on what groups and markers to search for.
                   also gives information on which is the master group and how the angle affects group.
    @min_obs -- minimum number of observations for transform required after finding it once.
    @n_avg -- number of times to average transform per observation.
    """
    global tf_listener
    if rospy.get_name() == '/unnamed':
        rospy.init_node('gripper_marker_calibration')
    tf_listener = tf.TransformListener()
    tf_listener.clear()

    group_info = {group:{} for group in calib_info}
    masterGraph = nx.DiGraph()
    ar_markers = []
    hydras = []
    ps_markers = []
    
    # Setup the groups
    for group in calib_info:
        group_info[group]["graph"] = nx.Graph()
        masterGraph.add_node(group_info[group]["graph"])
        masterGraph.node[group_info[group]["graph"]]['angle_scale'] = calib_info[group]['angle_scale']
        
        if calib_obs[group].get("master_group") is not None:
            masterGraph.node[group_info[group]["graph"]]["master"] = 1
            masterGraph.node[group_info[group]["graph"]]['angle_scale'] = 0

        group_info[group]["markers"] = calib_info["ar_markers"] + calib_info["hydras"] + calib_info["ps_markers"]
        group_info[group]["calib_info"] = calib_info[group]
        ar_markers.extend(calib_info["ar_markers"])
        hydras.extend(calib_info["hydras"])
        ps_markers.extend(calib_info["ps_markers"])


    if freq is not None:
        wait_time = 5
        print "Waiting for %f seconds before collecting data."%wait_time
        time.sleep(wait_time)

    sleeper = rospy.Rate(30)
    count = 0
    while True:
        if freq is None:
            raw_input(colorize("Iteration %d: Press return when ready to capture transforms."%count, "red", True))
        else: 
            print colorize("Iteration %d"%count, "red", True)
        
        avg_tfms = {}
        for j in xrange(n_avg):
            print colorize('\tGetting averaging transform : %d of %d ...'%(j,n_avg-1), "blue", True)
            

            tfms = {}
            tfms.update(get_ar_transforms(ar_markers, parent_frame))
            tfms.update(get_hydra_transforms(hydras, parent_frame))
            tfms.update(get_phasespace_transforms(ps_markers, parent_frame))
            
            pot_angle = get_potentiometer_angle()
                        
            for marker in tfms:
                if marker not in avg_tfms:
                    avg_tfms[marker] = []
                avg_tfms[marker].append(tfms[marker])
            
            sleeper.sleep()
        
        for marker in avg_tfms:
            avg_tfms[marker] = utils.avg_transform(avg_tfms[marker])

        update_groups_from_observations(masterGraph, group_info, avg_tfms, pot_angle)

        count += 1
        if is_ready(masterGraph, group_info, min_obs):
            if freq:
                break
            elif not yes_or_no("Enough data has been gathered. Would you like to gather more data anyway?"):
                break

    print "Finished gathering data in %d iterations."%count
    print "Calibrating for optimal transforms between markers..."
    G_opt = compute_relative_transforms (masterGraph, group_info, min_obs=min_obs)
    print "Finished calibrating."
    return G_opt
