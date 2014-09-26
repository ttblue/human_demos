# analyze the precision, and also plot the registration result.

import os, h5py, time, os.path as osp
import cPickle, pickle
import numpy as np, numpy.linalg as nlg
import math

import scipy
from scipy.spatial import *
import scipy.spatial.distance as ssd

import IPython as ipy

import openravepy, trajoptpy

from hd_rapprentice import registration, animate_traj, \
     plotting_openrave, task_execution, resampling, \
     ropesim_floating, rope_initialization, tps_registration
#from hd_rapprentice_old import registration as registration_old
from hd_rapprentice.registration import ThinPlateSpline, Affine, Composition
from hd_rapprentice.tps_registration import tps_segment_registration
from hd_utils import clouds, math_utils as mu, cloud_proc_funcs
#from hd_utils.pr2_utils import get_kinect_transform
from hd_utils.colorize import *
from hd_utils.utils import avg_transform
from hd_utils import transformations
from hd_utils.defaults import demo_files_dir, hd_data_dir,\
        ar_init_dir, ar_init_demo_name, ar_init_playback_name, \
        tfm_head_dof, tfm_bf_head, cad_files_dir, init_state_perturbs_dir
from hd_utils import math_utils
from knot_classifier import calculateCrossings, calculateMdp, isKnot, remove_crossing, pairs_to_dict, cluster_points
from knot_predictor import *
from hd_rapprentice import plotting_plt


import caffe
from caffe.proto import caffe_pb2

from joblib import Parallel, delayed
import argparse
from os import listdir
import cPickle as cp

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--data_folder", help="folder of data to be analyzed")
parser.add_argument("--use_ar_init", action="store_true")


args = parser.parse_args()

def plot_cb_gen(output_prefix, y_color, plot_color=1, proj_2d=1):
    def plot_cb(x_nd, y_md, x_color, f, s_cloud_id, tps_type):
        z_intercept = np.mean(x_nd[:,2])
        # Plot with color
        if plot_color:
            plotting_plt.plot_tps_registration(x_nd, y_md, f, res = (.1, .1, .12), x_color = x_color, y_color = y_color, proj_2d=proj_2d, z_intercept=z_intercept)
        else:
            plotting_plt.plot_tps_registration(x_nd, y_md, f, res = (.3, .3, .12), proj_2d=proj_2d, z_intercept=z_intercept)
        # save plot to file
        if output_prefix is not None:
            plt.savefig(osp.join(output_prefix, s_cloud_id + '_' + tps_type + '.png'))
    return plot_cb


def main():
    task_dir = osp.join(demo_files_dir, args.demo_type)
    demo_h5_file = osp.join(task_dir, args.demo_type + ".h5")
    data_files = []
    for fname in os.listdir(osp.join(task_dir, args.data_folder)):
        if not osp.isdir(osp.join(task_dir, args.data_folder, fname)):
            data_files.append(fname)
                        
    
    demofiles = h5py.File(demo_h5_file, "r")
    
    init_tfm = None
    if args.use_ar_init:
        # Get ar marker from demo:
        ar_demo_file = osp.join(hd_data_dir, ar_init_dir, ar_init_demo_name)
        with open(ar_demo_file,'r') as fh: ar_demo_tfms = cPickle.load(fh)
        # use camera 1 as default
        ar_marker_cameras = [1]
        ar_demo_tfm = avg_transform([ar_demo_tfms['tfms'][c] for c in ar_demo_tfms['tfms'] if c in ar_marker_cameras])
    
        # Get ar marker for PR2:
        # default demo_file
        ar_run_file = osp.join(hd_data_dir, ar_init_dir, ar_init_playback_name)
        with open(ar_run_file,'r') as fh: ar_run_tfms = cPickle.load(fh)
        ar_run_tfm = ar_run_tfms['tfm']
    
        # transform to move the demo points approximately into PR2's frame
        # Basically a rough transform from head kinect to demo_camera, given the tables are the same.
        init_tfm = ar_run_tfm.dot(np.linalg.inv(ar_demo_tfm))
        init_tfm = tfm_bf_head.dot(tfm_head_dof).dot(init_tfm)

    
    
    tps_results = {}
    tps_segment_name = ""
    for data_file in data_files:
        tps_results[data_file] = cp.load(open(osp.join(task_dir, args.data_folder, data_file)))
        if "segment" in data_file:
            tps_segment_name = data_file
            
    
    query_demo_keys = tps_results[tps_segment_name]["queries"]
    dataset_demo_keys = tps_results[tps_segment_name]["datasets"]
        
    # get the order
    sorted_indices_costs = {}
    for data_file in data_files:
        tps_result = tps_results[data_file]["results"]
        costs = []
        for key in query_demo_keys:
            cost = tps_result[key]["costs"]
            costs.append(cost)
                
        costs = np.asarray(costs)
        
        indices = np.argsort(costs, axis=1)
        sorted_indices_costs[data_file] = (indices, costs)
        
    
    n_queries = len(query_demo_keys)
    n_datasets = len(dataset_demo_keys)
    
        
    # compute precision/recall
    segment_sorted_indices_costs = ([], [])
    for i in range(n_queries):
        costs = sorted_indices_costs[tps_segment_name][1][i]
        indices = sorted_indices_costs[tps_segment_name][0][i]
        sorted_costs = costs[indices]
        id = np.where(sorted_costs==np.inf)[0]
        if len(id) != 0:
            indices = indices[:id[0]]
        segment_sorted_indices_costs[0].append(indices)
        segment_sorted_indices_costs[1].append(costs)
        
        

    
    tps_compare_statistics = {}
    for data_file in data_files:
        if data_file == tps_segment_name:
            pass
        else:
            max_K = 100
            precision_matrix = np.zeros([n_queries, max_K])
            recall_matrix = np.zeros([n_queries, max_K])
            cost_ratio_matrix = np.zeros([n_queries, max_K])
            
            for i in range(n_queries):
                indices = sorted_indices_costs[data_file][0][i]
                costs = sorted_indices_costs[data_file][1][i]
                sorted_costs = costs[indices]

                segment_indices = segment_sorted_indices_costs[0][i]
                segment_costs = segment_sorted_indices_costs[1][i]
                sorted_segment_costs = segment_costs[segment_indices]
                
                
                #for k in range(n_datasets):
                for k in range(max_K):
                    indices_k = indices[:k+1]
                    costs_k = sorted_costs[:k+1]
                    intersect_k = np.intersect1d(indices_k, segment_indices)
                    precision = len(intersect_k) / float(k+1)
                    recall = len(intersect_k) / float(len(segment_indices))
                    
                    precision_matrix[i, k] = precision
                    recall_matrix[i, k] = recall
                    
            average_precision = np.mean(precision_matrix, axis=0)
            average_recall = np.mean(recall_matrix, axis=0)
            
            tps_compare_statistics[data_file] = (average_recall, average_precision, precision_matrix, recall_matrix)


    # save the registration image, for only the second best to top K
    query_clouds = {}
    dataset_clouds = {}
    for query_key in query_demo_keys:
        demo_xyz = np.asarray(demofiles[query_key[0]][query_key[1]]["downsampled_cloud_xyz"])
        if init_tfm is not None:
            demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        query_clouds[query_key] = demo_xyz
    for dataset_key in dataset_demo_keys:
        demo_xyz = np.asarray(demofiles[dataset_key[0]][dataset_key[1]]["downsampled_cloud_xyz"])
        if init_tfm is not None:
            demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        dataset_clouds[dataset_key] = demo_xyz
    
    for data_file in data_files:
        for query_key in query_demo_keys:
            for dataset_key_id in range(n_datasets):
                dataset_key = dataset_demo_keys[dataset_key_id]
                query_cloud = query_clouds[query_key]
                dataset_cloud = dataset_clouds[dataset_key]
                data_filename = os.path.splitext(data_file)[0]
                if not osp.exists(osp.join(task_dir, args.data_folder, data_filename, query_key[0]+"_"+query_key[1])):
                    os.makedirs(osp.join(task_dir, args.data_folder, data_filename, query_key[0]+"_"+query_key[1]))
                plot_fn = plot_cb_gen(osp.join(task_dir, args.data_folder, data_filename, query_key[0]+"_"+query_key[1]), (0,0,1,1))
                 
                f = tps_results[data_file]["results"][query_key]["fs"][dataset_key_id]
                if f != None:
                    plot_fn(query_cloud, dataset_cloud, (1,0,0,1), f, dataset_key[0]+"_"+dataset_key[1], data_file)
                 
           
    color_set = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w'] 
    plt.figure()
    for index, compare_case in zip(range(len(tps_compare_statistics)), tps_compare_statistics.keys()):
        average_recall = tps_compare_statistics[compare_case][0]
        average_precision = tps_compare_statistics[compare_case][1]
        #plt.plot(average_recall, average_precision, color_set[index], label=compare_case)
        plt.plot(range(len(average_precision)), average_precision, color_set[index], label=compare_case)
        plt.legend(loc='upper right')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('tps compare result')
    plt.show()       
    
    import IPython
    IPython.embed()
            

                    
 
                    
                                
            
        
        
    
        
        
        
        
    
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    main()
