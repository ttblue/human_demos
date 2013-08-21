import numpy as np, numpy.linalg as nlg
import scipy as scp, scipy.optimize as sco

import networkx as nx
import itertools


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
    
    for i,j in itertools.combinations(range(len(ids)), 2):
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

    
def compute_relative_transforms (G, num_markers, min_obs=5):
    """
    Takes in a transform graph @G such that it has enough data to begin calibration (is_ready(G) returns true).
    Optimizes and computes final relative transforms between all nodes (markers).
    Returns a graph final_G with the same edges and final transforms computed.
    """
    
    assert (is_ready(G, num_markers=num_markers, min_obs=min_obs))
    
    