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


def plot_cb_gen(output_prefix, y_color, label_colors, plot_color=1, proj_2d=1):
    def plot_cb(x_nd, y_md, x_labels, y_labels, x_color, f, s_cloud_id, tps_type):
        z_intercept = np.mean(x_nd[:,2])
        # Plot with color
        if plot_color:
            plotting_plt.plot_tps_registration(x_nd, y_md, x_labels, y_labels, f, res = (.1, .1, .12), x_color = x_color, y_color = y_color, label_colors = label_colors, proj_2d=proj_2d, z_intercept=z_intercept)
        else:
            plotting_plt.plot_tps_registration(x_nd, y_md, x_labels, y_labels, f, res = (.3, .3, .12), proj_2d=proj_2d, z_intercept=z_intercept)
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
    
    
    
    
    # save the registration image, for only the second best to top K
    
    label_colors = {0: 'g', 1: 'c', 2: 'b', 3: 'y'}
    query_clouds = {}
    dataset_clouds = {}
    query_labels = {}
    dataset_labels = {}
    for query_key in query_demo_keys:
        demo_xyz = np.asarray(demofiles[query_key[0]][query_key[1]]["downsampled_cloud_xyz"])
        if init_tfm is not None:
            demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        query_clouds[query_key] = demo_xyz
        query_labels[query_key] = np.asarray(demofiles[query_key[0]][query_key[1]]["learned_features"]["label"])
    for dataset_key in dataset_demo_keys:
        demo_xyz = np.asarray(demofiles[dataset_key[0]][dataset_key[1]]["downsampled_cloud_xyz"])
        if init_tfm is not None:
            demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        dataset_clouds[dataset_key] = demo_xyz
        dataset_labels[dataset_key] = np.asarray(demofiles[dataset_key[0]][dataset_key[1]]["learned_features"]["label"])
        
    labeled_ropes = {}
    manual_labels = {}
    
    for dataset_key in dataset_demo_keys:
        labeled_points = np.asarray(demofiles[dataset_key[0]][dataset_key[1]]["labeled_points"])
        dataset_cloud = dataset_clouds[dataset_key]
        depth_image = np.asarray(demofiles[dataset_key[0]][dataset_key[1]]["depth"])
        depth_xyz = clouds.depth_to_xyz(depth_image)
        labeled_rope = np.empty((len(labeled_points),3))
        
        labels = []
        for i in range(len(labeled_points)):
            (x,y,c) = labeled_points[i,:]
            if i == 0 or i == len(labeled_points) - 1:
                labels.append(1)
            elif c == -1 or c == 1:
                labels.append(3)
            else:
                labels.append(2)
                
            labeled_rope[i] = depth_xyz[y,x]
            
        if init_tfm is not None:
            labeled_rope = labeled_rope.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
            
        labeled_ropes[dataset_key] = labeled_rope
        manual_labels[dataset_key] = labels

    
        
        
#    for dataset_key in dataset_demo_keys:                
#        labeled_rope = labeled_ropes[dataset_key]
#        labels = manual_labels[dataset_key]
#        dataset_cloud = dataset_clouds[dataset_key]
#        
#        colors = []
#        for i in range(len(labeled_rope)):
#            label = labels[i]
#            colors.append(label_colors[label])
#        
#        plt.ion()
#        fig = plt.figure('2d projection plot')
#        fig.clear()
#        plt.subplot(111, aspect='equal')
#        plt.axis('off')
#        plt.scatter(dataset_cloud[:, 0], dataset_cloud[:, 1], edgecolors=(1,0,0,1), marker='o', facecolors='none', s=50)
#        plt.scatter(labeled_rope[:,0], labeled_rope[:,1], c=colors, marker='+', s=70, linewidths=3)
#        plt.draw()
#        
#        plt.savefig(osp.join(osp.join(task_dir, args.data_folder, "segment", dataset_key[0]+"_"+dataset_key[1]+'.png')))
    
    
    
    
        
    # get the order
    sorted_indices_costs = {}
    for data_file in data_files:
        tps_result = tps_results[data_file]["results"]
        costs = []
        for key in query_demo_keys:
            cost = np.asarray(tps_result[key]["costs"])
            costs.append(cost)
            
        indices = []
        for cost in costs:
            index = np.argsort(cost)
            sorted_cost = cost[index]
            id = np.where(sorted_cost == np.inf)[0]
            if len(id) != 0:
                index = index[:id[0]]
            indices.append(index)
            
        sorted_indices_costs[data_file] = (indices, costs)
            
        
    
    n_queries = len(query_demo_keys)
    n_datasets = len(dataset_demo_keys)
    


    
        
    # compute precision/recall
    segment_sorted_indices_costs = sorted_indices_costs[tps_segment_name]
        
        
    
    tps_compare_statistics = {}
    for data_file in data_files:
        if data_file == tps_segment_name:
            pass
        else:
            #max_K = 100
            max_K = n_datasets
            precision_matrix = np.zeros([n_queries, max_K])
            recall_matrix = np.zeros([n_queries, max_K])
            
           
            
            for i in range(n_queries):
                indices = sorted_indices_costs[data_file][0][i]
                costs = sorted_indices_costs[data_file][1][i]
                sorted_costs = costs[indices]
                n_indices = len(indices)

                segment_indices = segment_sorted_indices_costs[0][i]
                segment_costs = segment_sorted_indices_costs[1][i]
                sorted_segment_costs = segment_costs[segment_indices]
                
                
                for k in range(max_K):
                    
                    effective_k = min(k, n_indices-1)
                    indices_k = indices[:effective_k+1]
                    costs_k = sorted_costs[:effective_k+1]
                    intersect_k = np.intersect1d(indices_k, segment_indices)
                    precision = len(intersect_k) / float(effective_k+1)
                    recall = len(intersect_k) / float(len(segment_indices))
                    
                    precision_matrix[i, k] = precision
                    recall_matrix[i, k] = recall
                    
            cost_sum = np.zeros([n_queries])
            for i in range(n_queries):
                segment_indices = segment_sorted_indices_costs[0][i]
                indices = np.intersect1d(sorted_indices_costs[data_file][0][i], segment_indices)
                costs = sorted_indices_costs[data_file][1][i][indices]
                cost_sum[i] = np.sum(costs)
                
                
            reg_error = np.zeros([n_queries])
            for query_id in range(n_queries):
                query_key = query_demo_keys[query_id]
                fs = tps_results[data_file]["results"][query_key]["fs"]
                fsegment = tps_results[tps_segment_name]["results"][query_key]["fs"]
                segment_indices = segment_sorted_indices_costs[0][query_id]
                query_rope = labeled_ropes[query_key]
                
                
                for dataset_id in segment_indices:
                    dataset_key = dataset_demo_keys[dataset_id]
                    # dataset_rope = labeled_ropes[dataset_key]
                    f_groundtruth = fsegment[dataset_id]
                    f = fs[dataset_id]
                    if f == None: continue
                    warped_groundtruth = f_groundtruth.transform_points(query_rope)
                    warped = f.transform_points(query_rope)
                    diffs = np.sum(np.abs(warped_groundtruth-warped)**2, axis=-1)**(1./2)
                    diff = np.sum(diffs)
                    reg_error[query_id] += diff

                    
            average_precision = np.mean(precision_matrix, axis=0)
            average_recall = np.mean(recall_matrix, axis=0)
            
                
            
            tps_compare_statistics[data_file] = (average_recall, average_precision, precision_matrix, recall_matrix, reg_error, cost_sum)


    
    for data_file in data_files:
        # if data_file != "result_deep_basic_use_ar_2_label.cp": continue
        # if data_file != "result_deep_basic_use_ar_2_label_new.cp": continue
        # if data_file != "result_deep_basic_use_ar_not_use_rough_2_label_score.cp": continue
        # if data_file != "result_deep_basic_use_ar_not_use_rough_2_label_fc8.cp": continue
        # if data_file != "result_deep_basic_use_ar_not_use_rough_2_fc8.cp": continue
        # if not (data_file == "result_basic_use_ar_2.cp" or data_file == "result_deep_basic_use_ar_2.cp"): continue
        # if data_file != "result_deep_basic_use_ar_2.cp": continue
        # if not ("lc" in data_file): continue 
        continue
        for query_key in query_demo_keys:
            for dataset_key_id in range(n_datasets):
                dataset_key = dataset_demo_keys[dataset_key_id]
                query_cloud = query_clouds[query_key]
                dataset_cloud = dataset_clouds[dataset_key]
                query_label = query_labels[query_key]
                dataset_label = dataset_labels[dataset_key]
                data_filename = os.path.splitext(data_file)[0]
                if not osp.exists(osp.join(task_dir, args.data_folder, data_filename, query_key[0]+"_"+query_key[1])):
                    os.makedirs(osp.join(task_dir, args.data_folder, data_filename, query_key[0]+"_"+query_key[1]))
                plot_fn = plot_cb_gen(osp.join(task_dir, args.data_folder, data_filename, query_key[0]+"_"+query_key[1]), (0,0,1,1), label_colors)
                 
                f = tps_results[data_file]["results"][query_key]["fs"][dataset_key_id]
                
                if f != None:
                    plot_fn(query_cloud, dataset_cloud, query_label, dataset_label, (1,0,0,1), f, dataset_key[0]+"_"+dataset_key[1], data_file)
                 
    #color_set = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#E5E5E5'] 
    plot_setting_set = [('r', '-'), ('g', '-'), ('b', '-'), ('c', '-'), ('m', '-'), ('y', '-'), ('k', '-'), ('r', '--'), ('g', '--'), ('b', '--'), ('c', '--'), ('m', '--'), ('y', '--'), ('k', '-')]
    
    reordered_tps_compare_statistics_keys = [None] * len(tps_compare_statistics.keys())
    reordered_end_id = 7
    for key in tps_compare_statistics.keys():
        if 'result_basic' in key:
            reordered_tps_compare_statistics_keys[0] = (key, "original TPS-RPM")
        elif 'score_fc8' in key:
            reordered_tps_compare_statistics_keys[4] = (key, "label+fc8 TPS-RPM")
        elif 'score_fc7' in key:
            reordered_tps_compare_statistics_keys[5] = (key, "label+fc7 TPS-RPM")
        elif 'fc8_fc7' in key:
            reordered_tps_compare_statistics_keys[6] = (key, "fc7+fc8 TPS-RPM")
        elif 'score' in key:
            reordered_tps_compare_statistics_keys[1] = (key, "label TPS-RPM")
        elif 'fc8' in key:
            reordered_tps_compare_statistics_keys[2] = (key, "fc8 TPS-RPM")
        elif 'fc7' in key:
            reordered_tps_compare_statistics_keys[3] = (key, "fc7 TPS-RPM")
        else:
            reordered_tps_compare_statistics_keys[reordered_end_id] = (key, key)
            reordered_end_id += 1


            
                            
    plt.figure(figsize=(10,5))
    #for index, compare_case in zip(range(len(tps_compare_statistics)), tps_compare_statistics.keys()):
    for index in range(len(tps_compare_statistics)):
        compare_case = reordered_tps_compare_statistics_keys[index][0]
        compare_case_show_name = reordered_tps_compare_statistics_keys[index][1]
        average_recall = tps_compare_statistics[compare_case][0]
        average_precision = tps_compare_statistics[compare_case][1]
        interpolated_precision = []
        for i in range(len(average_precision)):
            interpolated_precision.append(max(average_precision[i:]))
        plt.plot(average_recall, interpolated_precision, plot_setting_set[index][0], linestyle=plot_setting_set[index][1], label=compare_case_show_name)
        plt.legend(loc='upper right')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.show()       

    
    import IPython
    IPython.embed()
    
    plt.figure(figsize=(10,5))
    #for index, compare_case in zip(range(len(tps_compare_statistics)), tps_compare_statistics.keys()):
    for index in range(len(tps_compare_statistics)):
        compare_case = reordered_tps_compare_statistics_keys[index][0]
        compare_case_show_name = reordered_tps_compare_statistics_keys[index][1]
        average_recall = tps_compare_statistics[compare_case][0]
        average_precision = tps_compare_statistics[compare_case][1]
        interpolated_precision = []
        for i in range(len(average_precision)):
            interpolated_precision.append(max(average_precision[i:]))
        plt.plot(range(len(interpolated_precision)), interpolated_precision, plot_setting_set[index][0], linestyle=plot_setting_set[index][1], label=compare_case_show_name)
        plt.legend(loc='upper right')
    plt.xlabel('retrieval number')
    plt.ylabel('precision')
    plt.show()  
    
    
    for key in tps_compare_statistics.keys():
        print key
        print "   reg error"
        reg1 = (tps_compare_statistics[key][4][0] + tps_compare_statistics[key][4][3]) / (2 * n_datasets)
        reg2 = (tps_compare_statistics[key][4][1] + tps_compare_statistics[key][4][4]) / (2 * n_datasets)
        reg3 = (tps_compare_statistics[key][4][2] + tps_compare_statistics[key][4][5]) / (2 * n_datasets)
        print reg1, reg2, reg3
        print "   bend cost"
        cost1 = (tps_compare_statistics[key][5][0] + tps_compare_statistics[key][5][3]) / (2 * n_datasets)
        cost2 = (tps_compare_statistics[key][5][1] + tps_compare_statistics[key][5][4]) / (2 * n_datasets)
        cost3 = (tps_compare_statistics[key][5][2] + tps_compare_statistics[key][5][5]) / (2 * n_datasets)
        print cost1, cost2, cost3

        
    
    import IPython
    IPython.embed()
            

                    
 
                    
                                
            
        
        
    
        
        
        
        
    
    
    
    
    
    
    
    
    
    

if __name__ == "__main__":
    main()
