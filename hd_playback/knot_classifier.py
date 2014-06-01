#!/usr/bin/env python

import numpy as np
import os.path as osp
import h5py

#
# line segment intersection using vectors
# see Computer Graphics by F.S. Hill
#
def perp( a ) :
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b

# line segment a given by endpoints p1, p2
# line segment b given by endpoints p3, p4
# return
def seg_intersect(p1,p2,p3,p4) :
    p1=np.float32(p1)
    p2=np.float32(p2)
    p3=np.float32(p3)
    p4=np.float32(p4)
    numa = (p4-p3).dot(perp(p3-p1))
    numb = (p2-p1).dot(perp(p3-p1))
    denom = (p2-p1).dot(perp(p3-p4))
    if denom == 0:
        if numa==0 or numb==0: # coincident lines
            return (0.5,0.5)
        else: # parallel lines
            return None
    ua = (numa / denom)
    ub = (numb / denom)
    if ua >= 0 and ua <= 1 and ub >= 0 and ub <= 1:
        return (ua,ub)
    else:
        return None

def calculateIntersections(rope_nodes):
    """
    Takes in the nodes of a rope with n links.
    Returns the n x n matrix intersections, where intersections[i,j] = u_ij if link i intersects with link j at point pt_i, and intersections[i,j] = -1 otherwise.
    pt_i is the point on the line segment of link i with parameter u_ij.
    """
    intersections = -1*np.ones((rope_nodes.shape[0]-1, rope_nodes.shape[0]-1)) #n*n array of -1s
    for i_node in range(rope_nodes.shape[0]-1):
        for j_node in range(i_node+2,rope_nodes.shape[0]-1):
            intersect = seg_intersect(rope_nodes[i_node,:2], rope_nodes[i_node+1,:2], rope_nodes[j_node,:2], rope_nodes[j_node+1,:2])
            if intersect:
                intersections[i_node, j_node] = intersect[0]
                intersections[j_node, i_node] = intersect[1]
    return intersections

def calculateCrossings(rope_nodes, get_points=False, get_inds=False):
    """
    Returns a list of crossing patterns by following the rope nodes; +1 for overcrossings and -1 for undercrossings.
    """
    intersections = calculateIntersections(rope_nodes)
    crossings = []
    points = []
    inds = []
    links_to_cross_info = {} # Contains under-over crossing and crossing id info
    curr_cross_id = 1
    for i_link in range(intersections.shape[0]):
        j_links = sorted(range(intersections.shape[1]), key=lambda j_link: intersections[i_link,j_link])
        j_links = [j_link for j_link in j_links if intersections[i_link,j_link] != -1]
        for j_link in j_links:
            i_link_z = rope_nodes[i_link,2] + intersections[i_link,j_link] * (rope_nodes[i_link+1,2] - rope_nodes[i_link,2])
            j_link_z = rope_nodes[j_link,2] + intersections[j_link,i_link] * (rope_nodes[j_link+1,2] - rope_nodes[j_link,2])
            i_over_j = 1 if i_link_z > j_link_z else -1
            crossings.append(i_over_j)
            points.append(rope_nodes[i_link])
            inds.append(i_link)
            link_pair_id = (min(i_link,j_link), max(i_link,j_link))
            if link_pair_id not in links_to_cross_info:
                links_to_cross_info[link_pair_id] = []
            links_to_cross_info[link_pair_id].append((curr_cross_id, i_over_j))
            curr_cross_id += 1
    # make sure rope is closed -- each crossing should have an odd and even code
    rope_closed = True
    cross_pairs = set() # Set of tuples (a,b) where a and b are the indices of
                         # the over and under crossing-pair corresponding to the same crossing
    for cross_info in links_to_cross_info.values():
        if cross_info[0][0]%2 == cross_info[1][0]%2:
            rope_closed = False
        cross_pairs.add((cross_info[0][0], cross_info[1][0]))
    # dt_code = [0]*len(links_to_cross_info)
    # for cross_info in links_to_cross_info.values():
    # if cross_info[0][0]%2 == 0:
    # dt_code[cross_info[1][0]/2] = i_over_j * cross_info[0][0]
    # else:
    # dt_code[cross_info[0][0]/2] = i_over_j * cross_info[1][0]
    if get_points:
        return (crossings, points)
    if get_inds:
        return (crossings, inds)
    return (crossings, cross_pairs, rope_closed)

def crossingsToString(crossings):
    s = ''
    for c in crossings:
        if c == 1:
            s += 'o'
        elif c == -1:
            s += 'u'
    return s

def crossings_match(cross_pairs, top, s):
    # cross_pairs: Set of tuples (a,b) where a and b are the indices of
    # the over and under crossing-pair corresponding to the same crossing
    i = s.find(top) + 1 # Add 1, since the crossing pairs are 1-indexed
    if len(top) == 6:
        return (i,i+3) in cross_pairs and (i+1,i+4) in cross_pairs and \
               (i+2,i+5) in cross_pairs
    if len(top) == 8:
        return (i,i+5) in cross_pairs and (i+1,i+4) in cross_pairs and \
               (i+2,i+7) in cross_pairs and (i+3,i+6) in cross_pairs

def crossings_var_match(cross_pairs, top, s):
    i = s.find(top) + 1
    if len(top) == 8:
        return (i,i+4) in cross_pairs and (i+1,i+5) in cross_pairs and \
               (i+2,i+6) in cross_pairs and (i+3,i+7) in cross_pairs

#rope_nodes is an nx3 numpy array of the points of n control points of the rope
def isKnot(rope_nodes, rdm1=True):
    (crossings, cross_pairs, rope_closed) = calculateCrossings(rope_nodes)
    if rdm1: #apply Reidemeister move 1 where possible
        inds = np.sort(np.array(list(cross_pairs)).flatten())
        position = {}
        for i in inds:
            position[i] = np.where(inds==i)[0][0]
        for (o,u) in cross_pairs:
            if position[o] == position[u]-1:
                crossings[position[o]] = 'x'
                crossings[position[u]] = 'x'
        crossings = [i for i in crossings if i != "x"]
    s = crossingsToString(crossings)
    knot_topologies = ['uououo', 'uoouuoou']
    for top in knot_topologies:
        flipped_top = top.replace('u','t').replace('o','u').replace('t','o')
        if top in s and crossings_match(cross_pairs, top, s):
            return True
        if top[::-1] in s and crossings_match(cross_pairs, top[::-1], s):
            return True
        if flipped_top in s and crossings_match(cross_pairs, flipped_top, s):
            return True
        if flipped_top[::-1] in s and crossings_match(cross_pairs, flipped_top[::-1], s):
            return True

    if rope_closed:
        return False # There is no chance of it being a knot with one end
                      # of the rope crossing the knot accidentally

    # knot_topology_variations = ['ououuouo', 'ouoououu']
    # for top in knot_topology_variations:
    #     flipped_top = top.replace('u','t').replace('o','u').replace('t','o')
    #     if top in s and crossings_var_match(cross_pairs, top, s):
    #         return True
    #     if top[::-1] in s and crossings_var_match(cross_pairs, top[::-1], s):
    #         return True
    #     if flipped_top in s and crossings_var_match(cross_pairs, flipped_top, s):
    #         return True
    #     if flipped_top[::-1] in s and crossings_match(cross_pairs, flipped_top[::-1], s):
    #         return True

    return False


def matchTopology(xyz0, xyz1):
    (crossings0, cross_pairs0, _) = calculateCrossings(xyz0, get_points=True)
    (crossings1, cross_pairs1, _) = calculateCrossings(xyz1, get_points=True)
    if cross_pairs1 == cross_pairs0:
        return True
    else:
        print cross_pairs1, cross_pairs0
        return False

#returns a dictionary indexed by location, which stores the corresponding point at that location
def cluster_points(points, subset=None): 
#points is an array of coordinates in any-dimensional space
#subset is a subset of indices for points which must be matched to each other
    if subset == None:
        subset = range(len(points))
    pairs = {}
    for i in subset:
        if i not in pairs:
            min_dist = 50
            min_ind = None
            for j in subset:
                dist = np.linalg.norm(points[i]-points[j])
                if dist < min_dist and j != i:
                    min_dist = dist
                    min_ind = j
            pairs[i] = min_ind
            pairs[min_ind] = i
    return pairs


def remove_crossing(labeled_points, index): 
    """
    labeled_points is a numpy array of the form [[x0,y0,(z0),c0], [x1,y1,(z1),c1]... [xn,yn,(zn),cn]]
    where ci={-1,1} signifies a crossing and ci=0 indicates no crossing at that point.
    unlabel the index-th crossing and return the array with those labels set to 0.
    """
    import copy
    new_points = copy.copy(labeled_points)
    crossings_inds = [i for i in range(len(labeled_points)) if labeled_points[i][-1] != 0]
    pairs = cluster_points(new_points[:,:-1], crossings_inds)
    new_points[pairs[crossings_inds[index]]][-1] = 0
    new_points[crossings_inds[index]][-1] = 0
    return new_points

"""
Creates a dictionary of equivalent states.
equiv[cross_state] returns a list of all states equivalent to cross_state.
Two states (crossings patterns) A and B are equivalent if a segment with A
can transition into a state C that B can also transition into.
"""
def calculateMdp(hdf):
    stf = {}
    equiv = {}
    last = []
    for demo in hdf.keys():
        preceding = ()
        for seg in hdf[demo].keys():
            if hdf[demo][seg]['crossings'].shape[0] == 0:
                points = ()
            else:
                points = tuple(hdf[demo][seg]['crossings'][:,2])
            if preceding and preceding != points:
                if points in stf and preceding not in stf[points]:
                    stf[tuple(points)].append(preceding)
                elif points not in stf:
                    stf[tuple(points)] = [preceding]
            if seg == hdf[demo].keys()[-1]:
                last.append(tuple(points))
            preceding = points

    for state1 in stf.keys():
        for state2 in stf[state1]:
            for state in stf[state1]:
                if tuple(state2) in equiv:
                    equiv[tuple(state2)].append(state)
                else:
                    equiv[tuple(state2)] = [state]
    for state1 in last:
        for state2 in last:
            if state1 in equiv and state2 not in equiv[state1]:
                equiv[state1].append(state2)
            elif state1 not in equiv:
                equiv[state1] = [state2]
    return equiv

"""
Creates a dictionary of equivalent states, using labeled_points info if it exists.
equiv[cross_state] returns a list of all states equivalent to cross_state.
Two states (crossings patterns) A and B are equivalent if a segment with A
can transition into a state C that B can also transition into.
"""
def calculateMdp2(hdf):
    from do_task_floating import get_labeled_rope_demo
    stf = {}
    equiv = {}
    last = []
    for demo in hdf.keys():
        preceding = ()
        for seg in hdf[demo].keys():
            _, pattern = get_labeled_rope_demo(hdf[demo][seg], get_pattern=True)
            points = tuple(pattern)
            if preceding and preceding != points:
                if points in stf and preceding not in stf[points]:
                    stf[tuple(points)].append(preceding)
                elif points not in stf:
                    stf[tuple(points)] = [preceding]
            if seg == hdf[demo].keys()[-1]:
                last.append(tuple(points))
            preceding = points

    for state1 in stf.keys():
        for state2 in stf[state1]:
            for state in stf[state1]:
                if tuple(state2) in equiv:
                    equiv[tuple(state2)].append(state)
                else:
                    equiv[tuple(state2)] = [state]
    for state1 in last:
        for state2 in last:
            if state1 in equiv and state2 not in equiv[state1]:
                equiv[state1].append(state2)
            elif state1 not in equiv:
                equiv[state1] = [state2]
    return equiv


"""
Gives dictionary of topologies and corresponding demo/segments.
"""
def get_topology_dict(hdf):
    topos = {}
    for demo in hdf.keys():
        preceding = ()
        for seg in hdf[demo].keys():
            if hdf[demo][seg]['crossings'].shape[0] == 0:
                continue
            pattern = tuple([c[2] for c in hdf[demo][seg]['crossings'][:]])
            if pattern in topos.keys():
                topos[pattern].append((demo,seg))
            else:
                topos[pattern] = [(demo,seg)]
    return topos

"""
    todo:
        instead of is_knot, have match_topology classifer?
        learn knot_topologies arrays from SVM with labeled data
            or directly from point-cloud tracking data (tracking+calculateCrossings on demo pointclouds replaces labeling)
            keep track of locations of crossings as well as patterns, maybe for some sort of secondary TPS fit?
"""
