import numpy as np
import scipy.sparse as ss

def compute_cost(hydra_tfms, ar_tfms):

    """
    Returns the cost matrix given two lists of transforms by computing the Euclidean distance of the translation component in the tfm.
    """

    hydra_coords = [None] * len(hydra_tfms)
    ar_coords = [None] * len(hydra_tfms)

    for i in xrange(len(hydra_coords)):
        if hydra_tfms[i] != None:
            hydra_coords[i] = hydra_tfms[i][:3,3]
        if ar_tfms[i] != None:
            ar_coords[i] = ar_tfms[i][:3,3]
    
    direct_total_cost = 0
    count = 0
    for i in xrange(len(hydra_coords)):
        if ar_coords[i] != None and hydra_coords[i] != None:
            h = hydra_coords[i]
            a = ar_coords[i]
            diff = h - a
            direct_total_cost += np.linalg.norm(diff, 2)
            count += 1
    direct_avg_cost = direct_total_cost / count
    print "The average cost without DTW based on "+ str(count) + " matching transforms is " + str(direct_avg_cost) + " (m)"


    cost = np.zeros((len(hydra_tfms), len(ar_tfms)))

    for i in xrange(len(hydra_coords)):
        for j in xrange(len(ar_coords)):
            hydra_coord = hydra_coords[i]
            ar_coord = ar_coords[j]
            if hydra_coord == None or ar_coord == None:
                cost[i,j] = direct_avg_cost
            else:
                cost[i,j] = np.linalg.norm((hydra_coord - ar_coord), 2)
    #print cost.shape 
    return cost


def optimal_warp_path(cost):

    """
    Returns the optimal warping path using dynamic time warp with cost function as input.
    """
    m,n = cost.shape
    
    accum_cost = np.zeros((m, n))
    accum_cost[0, 0] = cost[0, 0]
    for i in xrange(m-1):
        accum_cost[i+1, 0] = accum_cost[i, 0] + cost[i+1, 0]
    for j in xrange(n-1):
        accum_cost[0, j+1] = accum_cost[0, j] + cost[0, j+1]

    for i in xrange(m-1):
        for j in xrange(n-1):
            accum_cost[i+1, j+1] = min(accum_cost[i, j], accum_cost[i, j+1], accum_cost[i+1, j]) + cost[i+1, j+1]

    print accum_cost
    reverse_path = []
    i = m-1
    j = n-1
    while (i, j) != (0, 0):
         reverse_path.append((i, j))
         if i == 0:
             j = j - 1
         elif j == 0:
             i = i - 1
         else:
             min_cost = min(accum_cost[i-1,j-1], accum_cost[i-1,j], accum_cost[i,j-1])
             print "min_cost is " + str(min_cost)
             if accum_cost[i-1,j-1] == min_cost:
                 i -= 1
                 j -= 1
                 print 'going diagonal'
             elif accum_cost[i-1,j] == min_cost:
                 i -= 1
                 print 'going up'
             else:
                 j -= 1 
                 'print going left'

    reverse_path.append((0, 0))

    print reverse_path
    return 0

