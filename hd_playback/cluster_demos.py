import h5py
import numpy as np
#from rapprentice import registration

from sklearn.cluster import spectral_clustering
import cPickle as pickle
import argparse
import cv2, hd_rapprentice.cv_plot_utils as cpu
import os.path as osp
import time

from hd_utils.defaults import demo_files_dir, similarity_costs_dir

np.set_printoptions(precision=6, suppress=True)

"""
Clusters based on costs in file.
"""

# Weights for different costs
weights = {}
weights['tps'] = 1.0
weights['traj'] = 0.2
weights['traj_f'] = 0.4


def get_costs (cfile):
    """
    Loads file with costs.
    """
    with open(cfile) as fh: return pickle.load(fh)


def get_name(demo_seg):
    """
    Gets shortened name for demo.
    """
    demo,seg = demo_seg
    try:
        return 'd%i'%int(demo[4:])+'s%i'%int(seg[3:])
    except:
        return demo+'-'+seg

def calc_sim(cost):
    """
    Calculates the cost of demo pair.
    """
    val = 0
    for c in cost:
        if c in weights: 
            if isinstance(cost[c], dict):
                val += sum(cost[c].values())*weights[c]
            else:
                val += cost[c]*weights[c]

    return val

def generate_sim_matrix (data, keys):
    """
    Generates the similarity matrix based on costs and weights.
    Costs and weights must have same dict keys.
    """
    cost_mat = np.zeros((len(keys), len(keys)))
    
    costs = data['costs']
    
    name_keys = {i:get_name(keys[i]) for i in keys}
    
    for i in xrange(len(keys)):
        print i
        for j in xrange(i+1,len(keys)):
            cost_mat[i,j] = calc_sim(costs[name_keys[i]][name_keys[j]])
    cost_mat = cost_mat + cost_mat.T # diagonal entries are 0. 

    return np.exp(-cost_mat)

def best_n_in_cluster(cluster, sm, n=None):
    """
    Returns the best n in cluster (closest to others).
    """
#     costs = np.zeros((len(cluster), len(cluster)))
#     for i in xrange(len(cluster)):
#         for j in xrange(len(cluster)):
#             costs[i][j] = sm[cluster[i]][cluster[j]]
    cluster_sm = sm[np.ix_(cluster, cluster)]
    sum_sm = np.sum(cluster_sm, axis=1)
    ranking = np.argsort(-sum_sm)
    
    if n is None: n = len(ranking)
    else: n = min(n, len(ranking))
    return [cluster[ranking[i]] for i in range(n)]


def rank_demos_in_cluster(clusters, sm):
    """
    Ranks all the demos in the clusters.
    """
    demo_cluster_rankings = {}
    idx = 0
    for i in clusters:
        cluster = clusters[i]
        if len(cluster) == 0: continue
        rankings = best_n_in_cluster(cluster, sm)
        demo_cluster_rankings[idx] = rankings
        idx += 1
    return demo_cluster_rankings

def cluster_and_rank_demos(sm, n_clusters, eigen_solver='arpack', assign_labels='discretize'):
    """
    Clusters demos based on similarity matrix.
    """
    labels = spectral_clustering(sm, n_clusters = n_clusters, eigen_solver=eigen_solver,assign_labels=assign_labels)
    clusters = {i:[] for i in xrange(n_clusters)}
    for i,l in enumerate(labels):
        clusters[l].append(i)

    # Maybe re-cluster large demos
    return rank_demos_in_cluster(clusters, sm)

def gen_h5_clusters (demo_type, cluster_data, keys, file_path=None):
    """
    Save .h5 file.
    """
    if file_path is not None:
        hdf = h5py.File(file_path,'w')
    else:
        cluster_path = osp.join(demo_files_dir, demo_type, demo_type+'_clusters.h5')
        hdf = h5py.File(cluster_path,'w')
    
    cgroup = hdf.create_group('clusters')
    for cluster in cluster_data:
        cgroup[str(cluster)] = cluster_data[cluster]
    
    kgroup = hdf.create_group('keys')
    for key in keys:
        kgroup[str(key)] = keys[key] 

def cluster_demos (demo_type, n_clusters, save_to_file=False, visualize=False, file_path=None):
    """
    Clusters and ranks demos.
    """
    demofile = h5py.File(osp.join(demo_files_dir, demo_type, demo_type+'.h5'), 'r')
    print "Loaded demo."
    cost_file = osp.join(demo_files_dir, demo_type, demo_type)+'.costs'
    costs = get_costs(cost_file)
    print "Loaded costs."
 
    seg_num = 0
    keys = {}
    for demo_name in demofile:
        if demo_name != "ar_demo":
            for seg_name in demofile[demo_name]:
                if seg_name != 'done':
                    keys[seg_num] = (str(demo_name), str(seg_name))
                    seg_num += 1
    
    print "Generating sim matrix."
    sm = generate_sim_matrix(costs, keys)
    
    print "Getting the cluster rankings"
    cdata = cluster_and_rank_demos(sm, n_clusters)
    
    if visualize:
        names = {i:[] for i in xrange(args.num_clusters)}
        images = {i:[] for i in xrange(args.num_clusters)}
        
        for i in cdata:
            names[i] = [get_name(keys[j]) for j in cdata[i]]
            images[i] = [np.asarray(demofile[keys[j][0]][keys[j][1]]["rgb"]) for j in cdata[i]]

        rows = []
        i = 0
        inc = True
        print "Press q to exit, left/right arrow keys to navigate"
        while True:
            if len(images[i]) == 0:
                if i == n_clusters-1: inc = False
                elif i == 0: inc = True
                if inc: i = min(i+1,n_clusters-1)
                else: i = max(i-1,0)                
                continue

            print "Label %i"%(i+1)
            print names[i]
            import math
            ncols = 7
            nrows = int(math.ceil(1.0*len(images[i])/ncols))
            row = cpu.tile_images(images[i], nrows, ncols)
            rows.append(np.asarray(row))
            cv2.imshow("clustering result", row)
            kb = cv2.waitKey()
            if kb == 1113939 or kb == 65363:
                i = min(i+1,args.num_clusters-1)
                inc = True
            elif kb == 1113937 or kb == 65361:
                i = max(i-1,0)
                inc = False
            elif kb == 1048689 or kb == 113:
                break
    
    
    if save_to_file:
        gen_h5_clusters(demo_type, cdata, keys, file_path)
    else:
        return cdata
    
def main(demo_type, n_clusters, num_seg=None):
    demofile = h5py.File(osp.join(demo_files_dir, demo_type, demo_type+'.h5'), 'r')
    print "Loaded file."
    iden = ''
    if num_seg is not None:
        iden = str(num_seg)
    cost_file = osp.join(similarity_costs_dir, demo_type)+iden+'.costs'
    
    costs = get_costs(cost_file)
    print "Got costs."
 
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

    ts = time.time()
    mat = generate_sim_matrix(costs, keys)
    print 'Time taken to generate sim matrix: %f'%(time.time() - ts)
    print mat
    
    ts = time.time()
    labels = spectral_clustering(mat, n_clusters = n_clusters, eigen_solver='arpack',assign_labels='discretize')
    print 'Time taken to cluster: %f'%(time.time() - ts)
    names = {i:[] for i in xrange(args.num_clusters)}
    images = {i:[] for i in xrange(args.num_clusters)}
    
    for i in xrange(len(labels)):
        label = labels[i]
        names[label].append(get_name(keys[i]))
        images[label].append(np.asarray(demofile[keys[i][0]][keys[i][1]]["rgb"]))



    rows = []
    i = 0
    inc = True
    print "Press q to exit, left/right arrow keys to navigate"
    while True:
        if len(images[i]) == 0:
            if i == n_clusters-1: inc = False
            elif i == 0: inc = True
            if inc: i = min(i+1,n_clusters-1)
            else: i = max(i-1,0)                
            continue

        print "Label %i"%(i+1)
        print names[i]
        import math
        ncols = 7
        nrows = int(math.ceil(1.0*len(images[i])/ncols))
        row = cpu.tile_images(images[i], nrows, ncols)
        rows.append(np.asarray(row))
        cv2.imshow("clustering result", row)
        kb = cv2.waitKey()
        if kb == 1113939 or kb == 65363:
            i = min(i+1,args.num_clusters-1)
            inc = True
        elif kb == 1113937 or kb == 65361:
            i = max(i-1,0)
            inc = False
        elif kb == 1048689 or kb == 113:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_type",help="Demo type.", type=str)
    parser.add_argument("--num_clusters",default=30, type=int)
    parser.add_argument("--num_segs", type=int, default=-1)
    parser.add_argument("--save", action="store_true", default=False)
    parser.add_argument("--visualize", action="store_true", default=False)
    args = parser.parse_args()

    if args.save:
        cluster_demos (args.demo_type, args.num_clusters, save_to_file=True, visualize=args.visualize)
    else:    
        if args.num_segs < 0:
            ns = None
        else:
            ns = args.num_segs
        main(args.demo_type, args.num_clusters, ns)
