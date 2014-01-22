import h5py
import numpy as np
#from rapprentice import registration

from joblib import Parallel, delayed
import scipy.spatial.distance as ssd
from sklearn.cluster import spectral_clustering
import cPickle as pickle
import argparse
import cv2, hd_rapprentice.cv_plot_utils as cpu
import os.path as osp
import time

from hd_rapprentice import registration
from hd_utils import clouds
from hd_utils.defaults import demo_files_dir, similarity_costs_dir

np.set_printoptions(precision=6, suppress=True)


"""
Clusters based on costs in file.
"""

def get_costs (cfile):
    with open(cfile) as fh: return pickle.load(fh)


def get_name(demo_seg):
    demo,seg = demo_seg
    try:
        return 'd%i'%int(demo[4:])+'s%i'%int(seg[3:])
    except:
        return demo+'-'+seg

def calc_sim(cost, weights):
    
    val = 0
    for c in cost:
        if c in weights: 
            if isinstance(cost[c], dict):
                val += sum(cost[c].values())*weights[c]
            else:
                val += cost[c]*weights[c]
    
    return val

def generate_sim_matrix (data, weights, keys):
    """
    Costs and weights must have same dict keys.
    """
    cost_mat = np.zeros((len(keys), len(keys)))
    
    costs = data['costs']
    
    name_keys = {i:get_name(keys[i]) for i in keys}
    
    for i in xrange(len(keys)):
        cost_mat[i,i] = 0.0
        for j in xrange(i+1,len(keys)):
            cost_mat[i,j] = calc_sim(costs[name_keys[i]][name_keys[j]], weights)
            cost_mat[j,i] = cost_mat[i,j] 

    return np.exp(-cost_mat)

def main(demo_type, n_clusters, num_seg=None):
    demofile = h5py.File(osp.join(demo_files_dir, demo_type, demo_type+'.h5'), 'r')
    
    iden = ''
    if num_seg is not None:
        iden = str(num_seg)
    cost_file = osp.join(similarity_costs_dir, demo_type)+iden+'.costs'
    
    costs = get_costs(cost_file)

    weights = {}    
    weights['tps'] = 0.5
    #weights['tps_scaled'] = 1.0
    weights['traj'] = 1.2
    weights['traj_f'] = 1.2
    #weights['traj_f_scaled'] = 0.5 
 
    seg_num = 0
    keys = {}
    done = False
    for demo_name in demofile:
        if demo_name != "ar_demo":
            for seg_name in demofile[demo_name]:
                if seg_name != 'done':
                    keys[seg_num] = (demo_name, seg_name)
                    seg_num += 1
                    if num_seg is not None and seg_num >= num_seg:
                        done = True
                        break
        if done:
            break

    mat = generate_sim_matrix(costs, weights,keys)
    print mat
    
    labels = spectral_clustering(mat, n_clusters = n_clusters, eigen_solver='arpack',assign_labels='discretize')
    names = {i:[] for i in xrange(args.num_clusters)}
    images = {i:[] for i in xrange(args.num_clusters)}
    
    for i in xrange(len(labels)):
        label = labels[i]
        names[label].append(get_name(keys[i]))
        images[label].append(np.asarray(demofile[keys[i][0]][keys[i][1]]["rgb"]))



    rows = []
    i = 0
    print "Press q to exit, left/right arrow keys to navigate"
    while True:
        print "Label %i"%(i+1)
        print names[i]
        import math
        ncols = 7
        nrows = int(math.ceil(1.0*len(images[i])/ncols))
        row = cpu.tile_images(images[i], nrows, ncols)
        rows.append(np.asarray(row))
        cv2.imshow("clustering result", row)
        kb = cv2.waitKey()
        if kb == 1113939:
            i = min(i+1,args.num_clusters-1)
        elif kb == 1113937:
            i = max(i-1,0)
        elif kb == 1048689:
            break
    return
    bigimg = cpu.tile_images(rows, len(rows), 50)
    cv2.imshow("clustering result", bigimg)
    print "press any key to continue"
    cv2.waitKey()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_type",help="Demo type.", type=str)
    parser.add_argument("--num_clusters", type=int)
    parser.add_argument("--num_segs", type=int, default=-1)
    args = parser.parse_args()

    if args.num_segs < 0:
        ns = None
    else:
        ns = args.num_segs
    main(args.demo_type, args.num_clusters, ns)
