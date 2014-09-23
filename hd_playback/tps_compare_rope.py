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


import caffe
from caffe.proto import caffe_pb2

from joblib import Parallel, delayed
import argparse

###################copy from do_task_merge start

CROSS_SECTION_SIZE = 4
DS_LEAF_SIZE = 0.01

def get_labeled_rope_demo(seg_group, get_pattern=False):
    labeled_points = seg_group["labeled_points"][:]
    depth_image = seg_group["depth"][:]
    labeled_rope = np.empty((len(labeled_points),4))
    depth_xyz = clouds.depth_to_xyz(depth_image)
    for i in range(len(labeled_rope)):
        (x,y,c) = labeled_points[i,:]
        labeled_rope[i,:3] = depth_xyz[y,x]
        labeled_rope[i,3] = c
        #labled_rope[i-1:i+2,2] += c*0.001 #move undercrossing points down a bit
    if not get_pattern:
        return labeled_rope
    else:
        if labeled_rope[-1][-1] != 0:
            print "remove last demo crossing (from end)"
            labeled_rope = remove_crossing(labeled_rope,-1)
        elif labeled_rope[0][-1] != 0:
            print "remove first demo crossing (from end)"
            labeled_rope = remove_crossing(labeled_rope,0)
        pattern = [pt[-1] for pt in labeled_rope if pt[-1]!=0]
        return labeled_rope, pattern
    
def observe_cloud(pts=None, radius=0.005, upsample=0, upsample_rad=1):
    """
    If upsample > 0, the number of points along the rope's backbone is resampled to be upsample points
    If upsample_rad > 1, the number of points perpendicular to the backbone points is resampled to be upsample_rad points, around the rope's cross-section
    The total number of points is then: (upsample if upsample > 0 else len(self.rope.GetControlPoints())) * upsample_rad

    Move to ropesim and/or ropesim_floating?
    """
    if upsample_rad > 1:
        # add points perpendicular to the points in pts around the rope's cross-section
        vs = np.diff(pts, axis=0) # vectors between the current and next points
        vs /= np.apply_along_axis(np.linalg.norm, 1, vs)[:,None]
        perp_vs = np.c_[-vs[:,1], vs[:,0], np.zeros(vs.shape[0])] # perpendicular vectors between the current and next points in the xy-plane
        perp_vs /= np.apply_along_axis(np.linalg.norm, 1, perp_vs)[:,None]
        vs = np.r_[vs, vs[-1,:][None,:]] # define the vector of the last point to be the same as the second to last one
        perp_vs = np.r_[perp_vs, perp_vs[-1,:][None,:]] # define the perpendicular vector of the last point to be the same as the second to last one
        perp_pts = []
        from openravepy import matrixFromAxisAngle
        for theta in np.linspace(0, 2*np.pi, upsample_rad, endpoint=False): # uniformly around the cross-section circumference
            for (center, rot_axis, perp_v) in zip(pts, vs, perp_vs):
                rot = matrixFromAxisAngle(rot_axis, theta)[:3,:3]
                perp_pts.append(center + rot.T.dot(radius * perp_v))
        pts = np.array(perp_pts)
    return pts


###################copy from do_task_merge end

def tps_segment_data(demofiles, demo_key, init_tfm):
    crossing_infos = []
    demo_xyzc = get_labeled_rope_demo(demofiles[demo_key[0]][demo_key[1]])
    for demo_xyzc_i in [demo_xyzc,demo_xyzc[::-1]]: # consider both original and reverse rope
        for num in [0,1]:        
            demo_xyz = demo_xyzc_i[:,:3]
    
            vs = np.diff(demo_xyz, axis=0)
            lengths = np.r_[0, np.apply_along_axis(np.linalg.norm, 1, vs)]
            summed_lengths = np.cumsum(lengths)
            assert len(lengths) == len(demo_xyz)
            demo_xyz = math_utils.interp2d(np.linspace(0, summed_lengths[-1], 67), summed_lengths, demo_xyz)
    
            if init_tfm is not None:
                demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
    
            ld = len(demo_xyz)
            # when generating points perpendicular to the backbone points, consider both clockwise and anti-clockwise
            demo_cloud1 = observe_cloud(demo_xyz, upsample_rad=CROSS_SECTION_SIZE)
            demo_cloud2 = observe_cloud(demo_xyz, upsample_rad=CROSS_SECTION_SIZE)
            demo_cloud2[:ld] = demo_cloud1[2*ld:3*ld]
            demo_cloud2[2*ld:3*ld] = demo_cloud1[:ld]
    
            _, demo_inds, _, demo_rope_closed = calculateCrossings(demo_xyz)
            demo_pattern = [int(pt[-1]) for pt in demo_xyzc_i if pt[-1]!=0]
            pair_dict = cluster_points(demo_xyz, subset=demo_inds)
            pair_inds = [(k, pair_dict[k]) for k in pair_dict.keys()]
            pairs = [(np.where(np.array(demo_inds)==i[0])[0][0], np.where(np.array(demo_inds)==i[1])[0][0]) for i in pair_inds]
            demo_pairs = []
            for pair in pairs:
                if (pair[0] < pair[1]):
                    demo_pairs.append((pair[0]+1, pair[1]+1))
            demo_pairs = set(demo_pairs)
    
            x_weights = 15*np.ones(len(demo_xyz)*CROSS_SECTION_SIZE)/(1.0*len(demo_xyz)*CROSS_SECTION_SIZE)
    
            assert(demo_cloud1.shape[1] == 3 and demo_cloud2.shape[1] == 3)
            if num == 0:
                crossing_infos.append((demo_xyz, demo_pattern, demo_inds, demo_pairs, demo_rope_closed, x_weights, demo_cloud1))
            if num == 1:
                crossing_infos.append((demo_xyz, demo_pattern, demo_inds, demo_pairs, demo_rope_closed, x_weights, demo_cloud2))
                
    return crossing_infos


def registration_cost_and_tfm_and_corr(xyz0, xyz1, num_iters=30, block_lengths=None):
    scaled_xyz0, src_params = registration.unit_boxify(xyz0)
    scaled_xyz1, targ_params = registration.unit_boxify(xyz1)
    f, _ = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, reg_init=10, reg_final = 0.01, 
            rad_init = .1, rad_final = .0005, rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=num_iters, plotting=True, block_lengths=block_lengths)
    cost = registration.tps_reg_cost(f)
    #cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    #g = registration.unscale_tps_3d(g, targ_params, src_params)
    f = registration.unscale_tps_3d(f, src_params, targ_params)
    return (cost, f, f.corr_nm) 

def registration_cost_and_tfm_and_corr_features(xyz0, xyz1, feature_costs, num_iters=30):
    scaled_xyz0, src_params = registration.unit_boxify(xyz0)
    scaled_xyz1, targ_params = registration.unit_boxify(xyz1)
    f, _ = registration.tps_rpm_bij_features(scaled_xyz0, scaled_xyz1, rot_reg = np.r_[1e-4, 1e-4, 1e-1], n_iter=10,
                                   reg_init=10, reg_final=0.4, rad_init=0.1, rad_final=0.005,
                                   outlierfrac=1e-2, feature_costs=feature_costs)
    
    cost = registration.tps_reg_cost(f)
    f = registration.unscale_tps_3d(f, src_params, targ_params)
    
    return (cost, f, f.corr_nm)



def tps_segment_core(demofiles, demo_key1, demo_key2, parallel, init_tfm = None):
    tps_segment_crossing_infos1 = tps_segment_data(demofiles, demo_key1, init_tfm)
    tps_segment_crossing_infos2 = tps_segment_data(demofiles, demo_key2, init_tfm)
    
    if parallel:
        results = Parallel(n_jobs=4)(delayed(tps_segment_registration)(info1[0:5], info2[0:5], info1[6], info2[6], np.eye(4), x_weights=info1[5], reg=0.01) 
                                     for info1 in tps_segment_crossing_infos1 for info2 in tps_segment_crossing_infos2)
        
        fs, corrs = zip(*results)

    else:
        fs = []
        corrs = []
        for info1 in tps_segment_crossing_infos1:
            for info2 in tps_segment_crossing_infos2:
                f, corr_nm = tps_segment_registration(info1[0:5], info2[0:5],
                                                      info1[6], info2[6],
                                                      np.eye(4), x_weights = info1[5], reg=0.01)
                
                fs.append(f)
                corrs.append(corr_nm)
        
    costs = []
    for f in fs:
        if f != None:
            costs.append(registration.tps_reg_cost(f))
        else:
            costs.append(np.inf)
            
    
    choice_ind = np.argmin(costs)
        
    return costs[choice_ind], fs[choice_ind], corrs[choice_ind]

def tps_segment_dataset(demofiles, query_demofiles, dataset_demofiles, parallel, init_tfm = None):
    query_demo_keys= []
    dataset_demo_keys = []
    
    for demo_name in demofiles:
        if demo_name in query_demofiles:
            for seg_name in demofiles[demo_name]:
                query_demo_keys.append((demo_name, seg_name))
        elif demo_name in dataset_demofiles:
            for seg_name in demofiles[demo_name]:
                dataset_demo_keys.append((demo_name, seg_name))
        else:
            pass
        
    query_results = {}
    for query_demo_key in query_demo_keys:
        results = []
        for dataset_demo_key in dataset_demo_keys:
            result = tps_segment_core(demofiles, query_demo_key, dataset_demo_key, parallel, init_tfm)
            results.append(result)
            
        costs, fs, corrs = zip(*results)
        choice_ind = np.argmin(costs)
        
        #query_results[query_demo_key] = (dataset_demo_keys[choice_ind], costs[choice_ind], fs[choice_ind], corrs[choice_ind])
        query_results[query_demo_key] = (choice_ind, costs, fs[choice_ind], corrs[choice_ind])
        print query_demo_key, costs[choice_ind]
    
    return query_results




def tps_basic_core(demofiles, demo_key1, demo_key2, init_tfm = None):
    demo_xyz1 = clouds.downsample(np.asarray(demofiles[demo_key1[0]][demo_key1[1]]["cloud_xyz"]), DS_LEAF_SIZE)
    demo_xyz2 = clouds.downsample(np.asarray(demofiles[demo_key2[0]][demo_key2[1]]["cloud_xyz"]), DS_LEAF_SIZE)
    
    if init_tfm is not None:
        demo_xyz1 = demo_xyz1.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        demo_xyz2 = demo_xyz2.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        
    result = registration_cost_and_tfm_and_corr(demo_xyz1, demo_xyz2)
    
    return result


def tps_basic_sequential(demofiles, query_demo_keys, dataset_demo_keys, init_tfm = None):
    query_results = {}
    for query_demo_key in query_demo_keys:
        query_result = []
        for dataset_demo_key in dataset_demo_keys:
            tps_result = tps_basic_core(demofiles, query_demo_key, dataset_demo_key, init_tfm)
            query_result.append(tps_result)
            
        costs, fs, corrs = zip(*query_result)
        choice_ind = np.argmin(costs)
        query_results[query_demo_key] = (choice_ind, costs, fs[choice_ind], corrs[choice_ind])
        print query_demo_key, costs[choice_ind]
        
    return query_results
    
    
def tps_basic_parallel(demofiles, query_demo_keys, dataset_demo_keys, init_tfm = None):
    query_demo_xyzs = []
    for demo_key in query_demo_keys:
        demo_xyz = clouds.downsample(np.asarray(demofiles[demo_key[0]][demo_key[1]]["cloud_xyz"]), DS_LEAF_SIZE)
        if init_tfm is not None:
            demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        query_demo_xyzs.append(demo_xyz)
        
    dataset_demo_xyzs = []
    for demo_key in dataset_demo_keys:
        demo_xyz = clouds.downsample(np.asarray(demofiles[demo_key[0]][demo_key[1]]["cloud_xyz"]), DS_LEAF_SIZE)
        if init_tfm is not None:
            demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        dataset_demo_xyzs.append(demo_xyz)
        
    all_results = Parallel(n_jobs=4, verbose=51)(delayed(registration_cost_and_tfm_and_corr)(query_demo_xyz, dataset_demo_xyz) for query_demo_xyz in query_demo_xyzs for dataset_demo_xyz in dataset_demo_xyzs)
    
    query_results = {}
    result_start_ind = 0
    num_dataset_demos = len(dataset_demo_keys)
    for query_demo_key in query_demo_keys:
        results = all_results[result_start_ind:result_start_ind + num_dataset_demos]
        costs, fs, corrs = zip(*results)
        choice_ind = np.argmin(costs)
        query_results[query_demo_key] = (choice_ind, costs, fs[choice_ind], corrs[choice_ind])
        print query_demo_key, costs[choice_ind]
        result_start_ind += num_dataset_demos
    
    return query_results    
 

def tps_basic_dataset(demofiles, query_demofiles, dataset_demofiles, parallel, init_tfm = None):
    
    query_demo_keys= []
    dataset_demo_keys = []
    
    for demo_name in demofiles:
        if demo_name in query_demofiles:
            for seg_name in demofiles[demo_name]:
                query_demo_keys.append((demo_name, seg_name))
        elif demo_name in dataset_demofiles:
            for seg_name in demofiles[demo_name]:
                dataset_demo_keys.append((demo_name, seg_name))
        else:
            pass
        
    if parallel:
        query_results = tps_basic_parallel(demofiles, query_demo_keys, dataset_demo_keys, init_tfm)
    else:
        query_results = tps_basic_sequential(demofiles, query_demo_keys, dataset_demo_keys, init_tfm)
        
    return query_results


def compute_label_costs(labels1, labels2):
    n_labels1 = len(labels1)
    n_labels2 = len(labels2)
    label_cost_matrix = np.zeros([n_labels1, n_labels2])
    for i in range(n_labels1):
        for j in range(n_labels2):
            if labels1[i] == labels2[j]:
                label_cost_matrix[i, j] = 0
            else:
                label_cost_matrix[i, j] = 1
                
    return label_cost_matrix
    

def tps_label_basic_core(demofiles, demo_key1, demo_key2, init_tfm = None):
    demo1 = demofiles[demo_key1[0]][demo_key1[1]]
    demo2 = demofiles[demo_key2[0]][demo_key2[1]]
    demo_xyz1 = np.asarray(demo["downsampled_cloud_xyz"])
    demo_xyz2 = np.asarray(demo["downsampled_cloud_xyz"])
    demo_rgb1 = demo1['rgb']
    demo_rgb2 = demo2['rgb']
    
    learned_labels1 = np.asarray(demo1['learned_label'])
    learned_labels2 = np.asarray(demo2['learned_label'])
    
    label_costs = compute_label_costs(learned_labels1, learned_labels2)
    
    if init_tfm is not None:
        demo_xyz1 = demo_xyz1.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        demo_xyz2 = demo_xyz2.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        
    result = registration_cost_and_tfm_and_corr_features(demo_xyz1, demo_xyz2, [label_costs])
    
    return result


def tps_label_basic_sequential(demofiles, query_demo_keys, dataset_demo_keys, init_tfm = None):
    query_results = {}
    for query_demo_key in query_demo_keys:
        query_result = []
        for dataset_demo_key in dataset_demo_keys:
            tps_result = tps_label_basic_core(demofiles, query_demo_key, dataset_demo_key, init_tfm)
            query_result.append(tps_result)
            
        costs, fs, corrs = zip(*query_result)
        choice_ind = np.argmin(costs)
        query_results[query_demo_key] = (choice_ind, costs, fs[choice_ind], corrs[choice_ind])
        print query_demo_key, costs[choice_ind]
        
    return query_results
    
    
def tps_label_basic_parallel(demofiles, query_demo_keys, dataset_demo_keys, init_tfm = None):
    query_demo_xyzs = []
    for demo_key in query_demo_keys:
        demo = demofiles[demo_key[0]][demo_key[1]]
        demo_xyz = np.asarray(demo["downsampled_cloud_xyz"])
        if init_tfm is not None:
            demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        query_demo_xyzs.append(demo_xyz)
        
    dataset_demo_xyzs = []
    for demo_key in dataset_demo_keys:
        demo = demofiles[demo_key[0]][demo_key[1]]
        demo_xyz = np.asarray(demo["downsampled_cloud_xyz"])
        if init_tfm is not None:
            demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        dataset_demo_xyzs.append(demo_xyz)
        
    query_labels = []
    for demo_key in query_demo_keys:
        demo = demofiles[demo_key[0]][demo_key[1]]
        demo_label = np.asarray(demo['learned_label'])
        query_labels.append(demo_label)
    
    dataset_labels = []
    for demo_key in dataset_demo_keys:
        demo = demofiles[demo_key[0]][demo_key[1]]
        demo_label = np.asarray(demo['learned_label'])
        dataset_labels.append(demo_label)
    
    all_results = Parallel(n_jobs=4, verbose=51)(delayed(registration_cost_and_tfm_and_corr_features)(query_demo_xyzs[i], dataset_demo_xyzs[j], [compute_label_costs(query_labels[i], dataset_labels[j])]) 
                                                 for i in range(len(query_demo_xyzs)) for j in range(len(dataset_demo_xyzs)))
    
    query_results = {}
    result_start_ind = 0
    num_dataset_demos = len(dataset_demo_keys)
    for query_demo_key in query_demo_keys:
        results = all_results[result_start_ind:result_start_ind + num_dataset_demos]
        costs, fs, corrs = zip(*results)
        choice_ind = np.argmin(costs)
        query_results[query_demo_key] = (dataset_demo_keys[choice_ind], costs[choice_ind], fs[choice_ind], corrs[choice_ind])
        print query_demo_key, query_results[query_demo_key][1]
        result_start_ind += num_dataset_demos
    
    return query_results    



def tps_label_basic_dataset(demofiles, query_demofiles, dataset_demofiles, net, parallel, recompute_label = False, init_tfm = None):
    
    query_demo_keys= []
    dataset_demo_keys = []
    
    for demo_name in demofiles:
        if demo_name in query_demofiles:
            for seg_name in demofiles[demo_name]:
                query_demo_keys.append((demo_name, seg_name))
        elif demo_name in dataset_demofiles:
            for seg_name in demofiles[demo_name]:
                dataset_demo_keys.append((demo_name, seg_name))
        else:
            pass

    if recompute_label:
        for demo_key in query_demo_keys:
            print "label", demo_key
            demo = demofiles[demo_key[0]][demo_key[1]]
            demo_xyz = clouds.downsample(np.asarray(demo["cloud_xyz"]), DS_LEAF_SIZE)
            demo_rgb = demo['rgb']
            demo_label, demo_score, demo_features, valid_mask = predictCrossing3D(demo_xyz, demo_rgb, net, demo_key[0] + "_" + demo_key[1])
            
            demo_xyz = demo_xyz[valid_mask]

            
            if "learned_label" in demo.keys():
                demo["learned_label"][()] = np.expand_dims(demo_label, axis=1) 
            else:
                demo["learned_label"] = demo_label    
                
            if "downsampled_cloud_xyz" in demo.keys():
                demo["downsampled_cloud_xyz"][()] = demo_xyz
            else:
                demo["downsampled_cloud_xyz"] = demo_xyz 
                
            if "learned_score" in demo.keys():
                demo["learned_score"][()] = demo_score      
            else:
                demo["learned_score"] = demo_score
                
            if not "learned_features" in demo.keys():
                demo.create_group("learned_features")
                
            features = demo["learned_features"]
            for feature_name in demo_features:
                if feature_name in features.keys():
                    features[feature_name][()] = demo_features[feature_name]
                else:
                    features[feature_name] = demo_features[feature_name]
                    
                
        for demo_key in dataset_demo_keys:
            print "label", demo_key
            demo = demofiles[demo_key[0]][demo_key[1]]
            demo_xyz = clouds.downsample(np.asarray(demo["cloud_xyz"]), DS_LEAF_SIZE)
            demo_rgb = demo['rgb']
            demo_label, demo_score, demo_features, valid_mask = predictCrossing3D(demo_xyz, demo_rgb, net, demo_key[0] + "_" + demo_key[1])
            
            if "learned_label" in demo.keys():
                demo["learned_label"][()] = np.expand_dims(demo_label, axis=1) 
            else:
                demo["learned_label"] = demo_label    
                
            if "downsampled_cloud_xyz" in demo.keys():
                demo["downsampled_cloud_xyz"][()] = demo_xyz
            else:
                demo["downsampled_cloud_xyz"] = demo_xyz 
                
            if "learned_score" in demo.keys():
                demo["learned_score"][()] = demo_score      
            else:
                demo["learned_score"] = demo_score
                
            if not "learned_features" in demo.keys():
                demo.create_group("learned_features")
                
            features = demo["learned_features"]
            for feature_name in demo_features:
                if feature_name in features.keys():
                    features[feature_name][()] = demo_features[feature_name]
                else:
                    features[feature_name] = demo_features[feature_name]      
                
                  
    demofiles.flush()

        
#    if parallel:
#        query_results = tps_label_basic_parallel(demofiles, query_demo_keys, dataset_demo_keys, init_tfm)
#    else:
#        query_results = tps_label_basic_sequential(demofiles, query_demo_keys, dataset_demo_keys, init_tfm)
#
#    return query_results

    return None
        
            

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--parallel", action="store_true")
parser.add_argument("--use_ar_init", action="store_true")
parser.add_argument("--tps_type", type=str, help="basic, segment, label_basic, label_segment")
parser.add_argument("--start_dataset", type=int, default=2)

parser.add_argument("--net_prototxt", help="File name for prototxt", type=str, default="")
parser.add_argument("--net_model", help="File name for learned model", type=str, default="")
parser.add_argument("--net_mean", help="File name for mean values", type=str, default="")
parser.add_argument("--recompute_label", help="Re-learn the label for each demo", action="store_true")


args = parser.parse_args()


def main():
    task_dir = osp.join(demo_files_dir, args.demo_type)
    demo_h5_file = osp.join(task_dir, args.demo_type + ".h5")
    
    demofiles = h5py.File(demo_h5_file, "r+")    
    
    if args.tps_type in ["label_basic", "label_segment"]:
        is_lenet = False
        if "lenet" == args.net_prototxt.split("_")[0]:
            is_lenet = True
        
        if is_lenet:
            net = caffe.Classifier(args.net_prototxt, args.net_model)
            net.set_phase_test()
            net.set_mode_gpu()
            net.set_input_scale('data', 1)
            net.set_channel_swap('data', (2,1,0))
        else:
            net = caffe.Classifier(args.net_prototxt, args.net_model)
            net.set_phase_test()
            net.set_mode_gpu()
            net.set_mean('data', np.load(args.net_mean))
            net.set_raw_scale('data', 255)
            net.set_channel_swap('data', (2,1,0))
        
        
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
        
    query_demofiles = []
    dataset_demofiles = []
    demo_ind = 0
    for demo_name in demofiles:
        if demo_ind < args.start_dataset:
            query_demofiles.append(demo_name)
        else:
            dataset_demofiles.append(demo_name) 
            
        demo_ind += 1   
                
                    
    if args.tps_type == "basic":     
        results = tps_basic_dataset(demofiles, query_demofiles, dataset_demofiles, args.parallel, init_tfm)
    elif args.tps_type == "segment":
        results = tps_segment_dataset(demofiles, query_demofiles, dataset_demofiles, args.parallel, init_tfm)
    elif args.tps_type == "label_basic":
        results = tps_label_basic_dataset(demofiles, query_demofiles, dataset_demofiles, net, args.parallel, args.recompute_label, init_tfm)

    
#    if args.use_ar_init:
#        use_ar = "use_ar"
#    else:
#        use_ar = "no_use_ar"
#        
#    result_filename = osp.join(task_dir, "result_" + args.tps_type + "_" + use_ar + "_" + str(args.start_dataset) + ".cp")
#    f = open(result_filename, 'wb')
#    pickle.dump([results, query_demofiles, dataset_demofiles], f)
    
    
            
if __name__ == "__main__":
    main()

            
            
            
            
            
            
            


