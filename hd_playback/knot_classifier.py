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

def calculateCrossings(rope_nodes):
    """
    Returns a list of crossing patterns by following the rope nodes; +1 for overcrossings and -1 for undercrossings.
    """
    intersections = calculateIntersections(rope_nodes)
    crossings = []
    points = []
    #links_to_cross_info = {}
    #curr_cross_id = 1
    for i_link in range(intersections.shape[0]):
        j_links = sorted(range(intersections.shape[1]), key=lambda j_link: intersections[i_link,j_link])
        j_links = [j_link for j_link in j_links if intersections[i_link,j_link] != -1]
        for j_link in j_links:
            i_link_z = rope_nodes[i_link,2] + intersections[i_link,j_link] * (rope_nodes[i_link+1,2] - rope_nodes[i_link,2])
            j_link_z = rope_nodes[j_link,2] + intersections[j_link,i_link] * (rope_nodes[j_link+1,2] - rope_nodes[j_link,2])
            i_over_j = 1 if i_link_z > j_link_z else -1
            crossings.append(i_over_j)
            points.append(rope_nodes[i_link])
#             link_pair_id = (min(i_link,j_link), max(i_link,j_link))
#             if link_pair_id not in links_to_cross_info:
#                 links_to_cross_info[link_pair_id] = []
#             links_to_cross_info[link_pair_id].append((curr_cross_id, i_over_j))
#             curr_cross_id += 1
#     # make sure rope is closed
#     dt_code = [0]*len(links_to_cross_info)
#     for cross_info in links_to_cross_info.values():
#         if cross_info[0][0]%2 == 0:
#             dt_code[cross_info[1][0]/2] = i_over_j * cross_info[0][0]
#         else:
#             dt_code[cross_info[0][0]/2] = i_over_j * cross_info[1][0]
    return crossings, np.array(points)

def crossingsToString(crossings):
    s = ''
    for c in crossings:
        if c == 1:
            s += 'o'
        elif c == -1:
            s += 'u'
    return s

#returns a dictionary indexed by location, returns the corresponding point at that location
def cluster_points(points):
    pairs = {}
    for i in range(len(points)):
        if i not in pairs:
            min_dist = 50
            min_ind = None
            for j in range(len(points)):
                dist = np.linalg.norm(points[i]-points[j])
                if dist < min_dist and j != i:
                    min_dist = dist
                    min_ind = j
            pairs[i] = min_ind
            pairs[min_ind] = i
    return pairs


#rope_nodes is an nx3 numpy array of the points of n control points of the rope
def isKnot(rope_nodes, rdm1=False):
    crossings, coords = calculateCrossings(rope_nodes)
    if rdm1: #apply Reidemeister move 1 where possible
        pairs = cluster_points(coords)
        for i in pairs:
            if i == pairs[i]-1:
                crossings[i] = "x"
                crossings[i+1] = "x"
        crossings = [i for i in crossings if i != "x"]
    s = crossingsToString(crossings)
    #knot_topologies = ['uououo', 'uoouuoou']
    knot_topologies = ['uououo']
    for top in knot_topologies:
        if top in s:
            return True
        if top[::-1] in s:
            return True
        flipped_top = top.replace('u','t').replace('o','u').replace('t','o')
        if flipped_top in s:
            return True
        if flipped_top[::-1] in s:
            return True
    return False


def matchTopology(rope_nodes, topology, demo_type):
    crossings, x = calculateCrossings(rope_nodes)
    s = crossingsToString(crossings)
    topologies = getTopologies(demo_type)
    h5filename = osp.join("/Users/George/Downloads", demo_type + '.h5')
    hdf = h5py.File(h5filename, 'r+')
    equiv = calculate_mdp(hdf);
    if s in equiv[topology]:
        return True
    else: return False


def getTopologies(demo_type):
    topologies = []
    h5filename = osp.join("/Users/George/Downloads", demo_type + '.h5')
    hdf = h5py.File(h5filename, 'r+')
    for demo in hdf.keys():
        for seg in hdf[demo].keys():
            topo = []
            for crossing in hdf[demo][seg]['crossings']:
                topo.append(crossing[2])
            if topo not in topologies:
                topologies.append(topo)
    topologies2 = []
    for topo in topologies:
        topologies2.append(crossingsToString(topo))
    return topologies2


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
    todo:
        instead of is_knot, have match_topology classifer?
        learn knot_topologies arrays from SVM with labeled data
            or directly from point-cloud tracking data (tracking+calculateCrossings on demo pointclouds replaces labeling)
            keep track of locations of crossings as well as patterns, maybe for some sort of secondary TPS fit?
"""
