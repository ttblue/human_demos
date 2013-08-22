import numpy as np, numpy.linalg as nlg
import scipy as scp, scipy.optimize as sco

import networkx as nx
import itertools

from hd_utils import utils

######### Tentative transform code ###############
def get_transform_ros(n_tfm, n_avg, freq=None):
    """
    Returns a tuple of two list of N_TFM transforms, each
    the average of N_AVG transforms
    corresponding to the AR marker with ID = MARKER and 
    the phasespace markers.
    """

    camera_frame = 'camera_depth_optical_frame'
    marker_frame = 'ar_marker_%d'%marker
    ps_frame = 'ps_marker_transform'

    tf_sub = tf.TransformListener()

    I_0 = np.eye(4)
    I_0[3,3] = 0
    
    ar_tfms    = []
    ph_tfms = []

    wait_time = 5
    print "Waiting for %f seconds before collecting data."%wait_time
    time.sleep(wait_time)

    sleeper = rospy.Rate(30)
    for i in xrange(n_tfm+1):
        if freq is None:
            raw_input(colorize("Transform %d of %d : Press return when ready to capture transforms"%(i, n_tfm), "red", True))
        else: 
            print colorize("Transform %d of %d."%(i, n_tfm), "red", True)
        ## transforms which need to be averaged.
        ar_tfm_avgs = []
        ph_tfm_avgs = []
        
        for j in xrange(n_avg):            
            print colorize('\tGetting averaging transform : %d of %d ...'%(j,n_avg-1), "blue", True)
            
            mtrans, mrot, ptrans, prot = None, None, None, None
            while ptrans == None or mtrans == None:
                mtrans, mrot = tf_sub.lookupTransform(camera_frame, marker_frame, rospy.Time(0))
                ptrans, prot = tf_sub.lookupTransform(ph.PHASESPACE_FRAME, ps_frame, rospy.Time(0))
                sleeper.sleep()

            ar_tfm_avgs.append(conversions.trans_rot_to_hmat(mtrans,mrot))
            ph_tfm_avgs.append(conversions.trans_rot_to_hmat(ptrans,prot))
            
        ar_tfm = utils.avg_transform(ar_tfm_avgs)
        ph_tfm = utils.avg_transform(ph_tfm_avgs)

#         print "\nar:"
#         print ar_tfm
#         print ar_tfm.dot(I_0).dot(ar_tfm.T)
#         print "h:"
#         print ph_tfm
#         print ph_tfm.dot(I_0).dot(ph_tfm.T), "\n"
                
        ar_tfms.append(ar_tfm)
        ph_tfms.append(ph_tfm)
        if freq is not None:
            time.sleep(1/freq)

        
    print "Found %i transforms. Calibrating..."%n_tfm
    Tas = ss.solve4(ar_tfms, ph_tfms)
    print "Done."
    
    T_cps = [ar_tfms[i].dot(Tas).dot(np.linalg.inv(ph_tfms[i])) for i in xrange(len(ar_tfms))]
    return utils.avg_transform(T_cps)


def update_graph_from_observations(G, marker_tfms):
    """
    Updates transform graph @G based on the observations @marker_tfms of AR marker transforms. 
    """
    ids = marker_tfms.keys()
    ids.sort()
    
    for i,j in itertools.combinations(xrange(len(ids)), 2):
        for k in [i,j]:
            if not G.has_node(k):
                G.add_node(k)
        if not G.has_edge(i,j):
            G.add_edge(i,j)
            G.edges[i][j]['transform_list'] = []
            G.edges[i][j]['n'] = 0
        
        Tij = nlg.inv(marker_tfms[i]).dot(marker_tfms[j])
        G.edges[i][j]['transform_list'].append(Tij)
        G.edges[i][j]['n'] += 1

            
def is_ready (G, num_markers, min_obs=5):
    """
    @num_markers is the total number of markers on the rigid object.
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

# Only for cliques --- not really relevant
#     def get_mat_from_x(X, n, i, j):
#         """
#         X is a vertical stack of variables for transformation matrices (12 variables each).
#         First n are Ti's.
#         Next n(n-1)/2 are Tij's, in lexicographic order of (i,j).
#         
#         Returns 3x4 matrix Tij
#         """
#         if j is None:
#             i_offset = 12*i
#             ij_offset = 0
#         else:
#             i_offset = 12*n
#             #  First term below is for edges (u,v) where u < i
#             # Second term below if for edges (i,v) where i < v < j
#             ij_offset = 12*[(n-1-(i-1)/2)*i + (j-i-2)]
# 
#         offset = i_offset + ij_offset
#         Xij = X[offset:offset+12]
#         
#         Tij = Xij.reshape([3,4], order='F')
#         Tij = np.r_[Tij, np.array([[0,0,0,1]])]
#         
#         return Tij

    # Index maps from nodes and edges in the optimizer variable X.
    # Also calculate the reverse map if needed.
    node_map, edge_map = {}, {}
    rev_node_map, rev_edge_map = {}, {}
    idx = 0

    for obj in G.nodes():
        node_map[obj] = idx
        rev_node_map[idx] = obj
        idx += 1
    for obj in G.edges():
        edge_map[obj] = idx
        rev_edge_map[idx] = obj
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
            con.append(nlg.norm)
            
            
        
    
def compute_relative_transforms (G, num_markers, min_obs=5):
    """
    Takes in a transform graph @G such that it has enough data to begin calibration (is_ready(G) returns true).
    Optimizes and computes final relative transforms between all nodes (markers).
    Returns a graph final_G with the same edges and final transforms computed.
    """
    
    assert (is_ready(G, num_markers=num_markers, min_obs=min_obs))
    
    for i,j in G.edges_iter():
        G[i][j]['avg_tfm'] = utils.avg_transform(G[i][j]['transform_list'])
        
