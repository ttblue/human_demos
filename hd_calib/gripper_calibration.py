#!/usr/bin/ipython -i
import numpy as np, numpy.linalg as nlg
import scipy.optimize as sco
import networkx as nx, networkx.algorithms as nxa
import itertools

import roslib; roslib.load_manifest("tf")
import rospy, tf

from hd_utils import utils, conversions
from hd_utils.yes_or_no import yes_or_no

tf_listener = None


def update_graph_from_observations(G, tfms):
    """
    Updates transform graph @G based on the observations @tfms, which is a dict from 
    marker ids (or names) to transforms all in the same coordinate frame. 
    """
    ids = tfms.keys()
    ids.sort()
    
    for i,j in itertools.combinations(ids, 2):
        for k in [i,j]:
            if not G.has_node(k):
                G.add_node(k)
        if not G.has_edge(i,j):
            G.add_edge(i,j)
            G.edges[i][j]['transform_list'] = []
            G.edges[i][j]['n'] = 0
        
        Tij = nlg.inv(tfms[i]).dot(tfms[j])
        G.edges[i][j]['transform_list'].append(Tij)
        G.edges[i][j]['n'] += 1

            
def is_ready (G, num_markers, min_obs=5):
    """
    @num_markers is the total number of markers/sensors on the rigid object.
    @min_obs is the minimum number of observations required for each relative transform, once one is seen. 
    
    Returns True when the graph has enough data to begin calibration, False otherwise.
    """
    if nx.is_connected(G) and G.number_of_nodes() == num_markers:
        for i,j in G.edges_iter():
            if G.edge[i][j]['n'] < min_obs:
                return False
        return True
    else:
        return False


def optimize_for_transforms (G):
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
    for obj in G.edges_iter():
        edge_map[obj] = idx
        rev_map[idx] = obj
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
            offset = edge_map[i,j]*12
        
        Xij = X[offset:offset+12]
        return Xij.reshape([3,4], order='F')


    def f_objective (X):
        """
        Objective function to make transforms close to average transforms.
        Sum of the norms of matrix differences between each Tij and 
        """        
        obj = 0
        for i,j in edge_map: 
            obj += nlg.norm(get_mat_from_x(X, i, j) - G[i][j]['avg_tfm'])
            
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
            con.append(nlg.norm(R.T.dot(R) - I3))
        
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
    
    (X, fx, _, _, _) = sco.fmin_slsqp(func=f_objective, x0=0, f_eqcons=f_constraints, iter=200, full_output=1)
    
    # Create output optimal graph and copy edge transforms
    G_opt = G.to_directed()
    for i,j in G.edges_iter():
        Tij = np.r_[get_mat_from_x(X, i, j),np.array([0,0,0,1])]
        Tji = nlg.inv(Tij)
        
        G_opt.edge[i][j] = {'tfm':Tij}
        G_opt.edge[j][i] = {'tfm':Tji}
        
    # Add all edges to make clique
    # Follow shortest path to get transform for edge
    for i,j in itertools.permutations(G_opt.nodes(), 2):
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

    return G_opt
            
    
    
def compute_relative_transforms (G, num_markers, min_obs=5):
    """
    Takes in a transform graph @G such that it has enough data to begin calibration (is_ready(G) returns true).
    Optimizes and computes final relative transforms between all nodes (markers).
    Returns a graph final_G with the all edges (clique) and final transforms stored in edges.
    """
    
    assert (is_ready(G, num_markers=num_markers, min_obs=min_obs))
    for i,j in G.edges_iter():
        G[i][j]['avg_tfm'] = utils.avg_transform(G[i][j]['transform_list'])
    
    G_opt = optimize_transforms(G)
    return G_opt


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
        except LookupException:
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


def create_graph_from_observations(parent_frame, num_markers, obs_info, min_obs=5, n_avg=5, freq=None):
    """
    Runs a loop till graph has enough data and user is happy with data.
    Or run with frequency specified until enough data is gathered.
    
    @parent_frame -- frame to get observations in.
    @num_markers -- total_number of markers.
    @obs_info -- dict with information on what markers to search for.
    @min_obs -- 
    """
    global tf_listener
    if rospy.get_name == '/unnamed':
        rospy.init_node('gripper_marker_calibration')
    tf_listener = tf.TransformListener()
    
    G = nx.Graph()

    # setup for different sensors
    ar_markers = obs_info.get('ar_markers')
    hydras = obs_info.get('hydras')
    ps_markers = obs_info.get('phasespace_markers')

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
            
            for marker in tfms:
                if marker not in avg_tfms:
                    avg_tfms[marker] = []
                avg_tfms[marker].append(tfms[marker])
            
            sleeper.sleep()
        
        for marker in avg_tfms:
            avg_tfms[marker] = utils.avg_transform(avg_tfms[marker])
        
        count += 1
        update_graph_from_observations(G, avg_tfms)
        if is_ready(G,num_markers,min_obs):
            if freq:
                break
            elif not yes_or_no("Enough data has been gathered. Would you like to gather more data anyway?"):
                break

    print "Finished gathering data in %d iterations."%count
    print "Calibrating for optimal transforms between markers..."
    G_opt = compute_relative_transforms (G, num_markers=num_markers, min_obs=min_obs)
    print "Finished calibrating."
    return G_opt
