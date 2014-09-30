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
            demo_pairs = list(demo_pairs)
    
            x_weights = 15*np.ones(len(demo_xyz)*CROSS_SECTION_SIZE)/(1.0*len(demo_xyz)*CROSS_SECTION_SIZE)
    
            assert(demo_cloud1.shape[1] == 3 and demo_cloud2.shape[1] == 3)
            if num == 0:
                crossing_infos.append((demo_xyz, demo_pattern, demo_inds, demo_pairs, demo_rope_closed, x_weights, demo_cloud1))
            if num == 1:
                crossing_infos.append((demo_xyz, demo_pattern, demo_inds, demo_pairs, demo_rope_closed, x_weights, demo_cloud2))
           
    return crossing_infos

def load_tps_segment_data(demofiles, demo_key, init_tfm):
    if init_tfm == None:
        crossing_infos_group = demofiles[demo_key[0]][demo_key[1]]["crossing_infos"]
    else:
        crossing_infos_group = demofiles[demo_key[0]][demo_key[1]]["crossing_infos_ar_init"]
        
    demo_xyz_list = np.asarray(crossing_infos_group["demo_xyz"])
    demo_pattern_list_ = np.asarray(crossing_infos_group["demo_pattern"])
    demo_pattern_list = []
    for i in range(len(demo_pattern_list_)):
        demo_pattern_list.append(list(demo_pattern_list_[i]))
    
    demo_inds_list = np.asarray(crossing_infos_group["demo_inds"])
    demo_pairs_list_ = np.asarray(crossing_infos_group["demo_pairs"])

    demo_pairs_list = []
    for i in range(len(demo_pairs_list_)):
        demo_pairs_ = demo_pairs_list_[i]
        demo_pairs = []
        for j in range(len(demo_pairs_)):
            demo_pairs.append((demo_pairs_[j][0], demo_pairs_[j][1]))
        demo_pairs_list.append(set(demo_pairs))

    
    demo_rope_closed_list = np.asarray(crossing_infos_group["demo_rope_closed"])
    x_weights_list = np.asarray(crossing_infos_group["x_weights"])
    demo_cloud_list = np.asarray(crossing_infos_group["demo_cloud"])
    

    crossing_infos2 = tps_segment_data(demofiles, demo_key, init_tfm)
    

    crossing_infos = []
    n = len(demo_xyz_list)
    for i in range(n):
        crossing_infos.append((demo_xyz_list[i], demo_pattern_list[i], demo_inds_list[i], demo_pairs_list[i], demo_rope_closed_list[i], x_weights_list[i], demo_cloud_list[i]))
 
    return crossing_infos


def registration_cost_and_tfm_and_corr(xyz0, xyz1, num_iters=10):
    scaled_xyz0, src_params = registration.unit_boxify(xyz0)
    scaled_xyz1, targ_params = registration.unit_boxify(xyz1)
    
    
    f, _ = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, reg_init=10, reg_final = 0.01, 
            rad_init = .1, rad_final = .0005, rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=num_iters, plotting=False)
    cost = registration.tps_reg_cost(f)
    #cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    #g = registration.unscale_tps_3d(g, targ_params, src_params)
    f = registration.unscale_tps_3d(f, src_params, targ_params)
    return (cost, f, f.corr_nm) 


def rotate_about_median(xyz, theta, median=None):
    """
    rotates xyz by theta around the median along the x, y dimensions
    """
    if median==None:
        median = np.median(xyz, axis=0)
    centered_xyz = xyz - median
    if np.shape(theta):
        r_mat = theta
        rotated_xyz = centered_xyz.dot(r_mat)
        rotated_xyz = rotated_xyz - np.median(rotated_xyz, axis=0) #recenter to avoid numerical issues - hacky
    else:
        r_mat = np.eye(3)
        r_mat[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        rotated_xyz = centered_xyz.dot(r_mat)
    new_xyz = rotated_xyz + median
    return new_xyz

def registration_cost(xyz0, xyz1, num_iters=10, block_lengths=None):
    scaled_xyz0, _ = registration.unit_boxify(xyz0)
    scaled_xyz1, _ = registration.unit_boxify(xyz1)
    f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, reg_init=10, reg_final = 0.01, 
                                    rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=num_iters, block_lengths=block_lengths)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    return cost

def rotations_from_ang(theta,median):
    if not np.shape(theta):
        rotation_matrix = np.eye(3)
        rotation_matrix_inv = np.eye(3)
        rotation_matrix[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        rotation_matrix_inv[0:2, 0:2] = np.array([[np.cos(-theta), -np.sin(-theta)],[np.sin(-theta), np.cos(-theta)]])
    else:
        rotation_matrix = theta; rotation_matrix_inv = np.linalg.inv(rotation_matrix)
    shift_from_origin = Affine(np.eye(3), median)
    rotate = Affine(rotation_matrix, np.zeros((3,)))
    unrotate = Affine(rotation_matrix_inv, np.zeros((3,)))
    shift_to_origin = Affine(np.eye(3), -1*median)
    rotate = Composition([shift_to_origin,rotate,shift_from_origin])
    unrotate = Composition([shift_to_origin,unrotate,shift_from_origin])
    return rotate, unrotate



def registration_cost_and_tfm_and_corr_features(xyz0, xyz1, label0, label1, feature_costs, feature_weights_initial, feature_weights_final, rough_init, num_iters=10):
    
    rad_angs = np.linspace(-np.pi+np.pi*(float(90)/180), np.pi,4)
    Rs = []
    costs = np.zeros(len(rad_angs))
    for i in range(len(rad_angs)):
        rotated_demo = rotate_about_median(xyz0, rad_angs[i])
        R, _ = registration.icp(rotated_demo, xyz1, 20)
        Rs.append(R)
        costs[i] = registration_cost(rotated_demo.dot(R.T), xyz1, 5)
    theta = rad_angs[np.argmin(costs)]
    R = Rs[np.argmin(costs)]
    
    theta_mat = np.eye(3)
    theta_mat[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

    theta_mat = theta_mat.dot(R.T)
    rotated_xyz0 = rotate_about_median(xyz0, theta_mat)
    
    
    
    scaled_xyz0, src_params = registration.unit_boxify(rotated_xyz0) # changed
    scaled_xyz1, targ_params = registration.unit_boxify(xyz1)
    
    
#    f,g,corr_nm = registration.tps_rpm_bij(scaled_x_nd, scaled_y_md, rot_reg = np.r_[1e-4, 1e-4, 1e-1], n_iter=10,
#                                   reg_init=10, reg_final=0.4, rad_init=0.1, rad_final=0.005,
#                                   outlierfrac=1e-8, vis_cost_xy=vis_cost_xy, return_corrnm = True,
#                                   plotting=0, plot_cb=None)
#                                   #plotting=plotting, plot_cb=plot_cb_bij_gen(None, proj_2d, bgr_to_rgb(xyzrgb0[:,-3:]), bgr_to_rgb(xyzrgb1[:,-3:])))

    f, _ = registration.tps_rpm_bij_features(scaled_xyz0, scaled_xyz1, label0, label1, rough_init, reg_init=10, reg_final = 0.001, #0.01, 
            rad_init = .1, rad_final = .0005, rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=num_iters, plotting=False, feature_costs=feature_costs, feature_weights_initial = feature_weights_initial, feature_weights_final = feature_weights_final)
    

    res_cost = f._cost_tuple[0]
    bend_cost = f._cost_tuple[3]
    f = registration.unscale_tps_3d(f, src_params, targ_params)
    corr_nm = f.corr_nm

    
    rotate, unrotate = rotations_from_ang(theta_mat,np.median(xyz0, axis=0))
    f = Composition([rotate, f])
    
    return (res_cost, bend_cost, f, corr_nm)



def tps_segment_core(demofiles, demo_key1, demo_key2, parallel, init_tfm = None):
    tps_segment_crossing_infos1 = load_tps_segment_data(demofiles, demo_key1, init_tfm)
    tps_segment_crossing_infos2 = load_tps_segment_data(demofiles, demo_key2, init_tfm)
    
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

def tps_segment_dataset(demofiles, query_demofiles, dataset_demofiles, parallel, re_precompute = False, init_tfm = None):
    query_demo_keys= []
    dataset_demo_keys = []
    
    for demo_name in demofiles:
        if demo_name in query_demofiles:
            for seg_name in demofiles[demo_name]:
                query_demo_keys.append((demo_name, seg_name))
        if demo_name in dataset_demofiles:
            for seg_name in demofiles[demo_name]:
                dataset_demo_keys.append((demo_name, seg_name))
        
    if re_precompute:
        for demo_name in demofiles:
            for seg_name in demofiles[demo_name]:
                demo = demofiles[demo_name][seg_name]
                demo_key = (demo_name, seg_name)
                crossing_infos = tps_segment_data(demofiles, demo_key, init_tfm)
                
                if init_tfm == None:
                    if not "crossing_infos" in demo.keys():
                        demo.create_group("crossing_infos")
                    crossing_infos_group = demo["crossing_infos"]
                else:
                    if not "crossing_infos_ar_init" in demo.keys():
                        demo.create_group("crossing_infos_ar_init")
                    crossing_infos_group = demo["crossing_infos_ar_init"]                    
                
                demo_xyz_list, demo_pattern_list, demo_inds_list, demo_pairs_list, demo_rope_closed_list, x_weights_list, demo_cloud_list = zip(*crossing_infos)
                
                if "demo_xyz" in crossing_infos_group.keys():
                    crossing_infos_group["demo_xyz"][()] = np.asarray(demo_xyz_list)
                else:
                    crossing_infos_group["demo_xyz"] = np.asarray(demo_xyz_list)
                                        
                if "demo_pattern" in crossing_infos_group.keys():
                    crossing_infos_group["demo_pattern"][()] = np.asarray(demo_pattern_list)
                else:
                    crossing_infos_group["demo_pattern"] = np.asarray(demo_pattern_list)                    
                
                if "demo_inds" in crossing_infos_group.keys():
                    crossing_infos_group["demo_inds"][()] = np.asarray(demo_inds_list)
                else:
                    crossing_infos_group["demo_inds"] = np.asarray(demo_inds_list)
                 
                if "demo_pairs" in crossing_infos_group.keys():
                    crossing_infos_group["demo_pairs"][()] = np.asarray(demo_pairs_list)
                else:
                    crossing_infos_group["demo_pairs"] = np.asarray(demo_pairs_list)       
                    
                if "demo_rope_closed" in crossing_infos_group.keys():
                    crossing_infos_group["demo_rope_closed"][()] = np.asarray(demo_rope_closed_list)
                else:
                    crossing_infos_group["demo_rope_closed"] = np.asarray(demo_rope_closed_list)     
                    
                if "x_weights" in crossing_infos_group.keys():
                    crossing_infos_group["x_weights"][()] = np.asarray(x_weights_list)
                else:
                    crossing_infos_group["x_weights"] = np.asarray(x_weights_list)      
                    
                if "demo_cloud" in crossing_infos_group.keys():
                    crossing_infos_group["demo_cloud"][()] = np.asarray(demo_cloud_list)
                else:
                    crossing_infos_group["demo_cloud"] = np.asarray(demo_cloud_list)  
                    
        demofiles.flush()
                      
        
        
    query_results = {}
    for query_demo_key in query_demo_keys:
        results = []
        for dataset_demo_key in dataset_demo_keys:
            print query_demo_key, dataset_demo_key
            result = tps_segment_core(demofiles, query_demo_key, dataset_demo_key, parallel, init_tfm)
            results.append(result)
            
        costs, fs, corrs = zip(*results)
        
        query_results[query_demo_key] = {"costs": costs, "fs": fs, "corrs": corrs}
        print query_demo_key
    
    return {"results": query_results, "queries": query_demo_keys, "datasets": dataset_demo_keys, "feature_types": []}    




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
        query_results[query_demo_key] = {"costs": costs, "fs": fs, "corrs": corrs}
        print query_demo_key
        
    return {"results": query_results, "queries": query_demo_keys, "datasets": dataset_demo_keys, "feature_types": []}    
    
    
def tps_basic_parallel(demofiles, query_demo_keys, dataset_demo_keys, init_tfm = None):
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
        
    #all_results = Parallel(n_jobs=4, verbose=51)(delayed(registration_cost_and_tfm_and_corr)(query_demo_xyz, dataset_demo_xyz) for query_demo_xyz in query_demo_xyzs for dataset_demo_xyz in dataset_demo_xyzs)
    all_results = Parallel(n_jobs=4, verbose=51)(delayed(registration_cost_and_tfm_and_corr)(query_demo_xyzs[i], dataset_demo_xyzs[j])
                                                 for i in range(len(query_demo_xyzs)) for j in range(len(dataset_demo_xyzs)))
    
    
    query_results = {}
    result_start_ind = 0
    num_dataset_demos = len(dataset_demo_keys)
    for query_demo_key in query_demo_keys:
        results = all_results[result_start_ind:result_start_ind + num_dataset_demos]
        costs, fs, corrs = zip(*results)
        query_results[query_demo_key] = {"costs": costs, "fs": fs, "corrs": corrs}
        print query_demo_key
        result_start_ind += num_dataset_demos
    
    return {"results": query_results, "queries": query_demo_keys, "datasets": dataset_demo_keys, "feature_types": []}    
 

def tps_basic_dataset(demofiles, query_demofiles, dataset_demofiles, parallel, init_tfm = None):
    
    query_demo_keys= []
    dataset_demo_keys = []
    
    for demo_name in demofiles:
        if demo_name in query_demofiles:
            for seg_name in demofiles[demo_name]:
                query_demo_keys.append((demo_name, seg_name))
        if demo_name in dataset_demofiles:
            for seg_name in demofiles[demo_name]:
                dataset_demo_keys.append((demo_name, seg_name))
        
    if parallel:
        query_results = tps_basic_parallel(demofiles, query_demo_keys, dataset_demo_keys, init_tfm)
    else:
        query_results = tps_basic_sequential(demofiles, query_demo_keys, dataset_demo_keys, init_tfm)
        
    return query_results



LABEL_COST_MATRIX_PARAM = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]])


def compute_label_costs(labels1, labels2):
    n_labels1 = len(labels1)
    n_labels2 = len(labels2)
    label_cost_matrix = np.zeros([n_labels1, n_labels2])
    for i in range(n_labels1):
        for j in range(n_labels2):
#            if labels1[i] == labels2[j]:
#                label_cost_matrix[i, j] = 0
#            else:
#                label_cost_matrix[i, j] = 1
            label_cost_matrix[i, j] = LABEL_COST_MATRIX_PARAM[labels1[i], labels2[j]]
                
    return label_cost_matrix

def compute_score_costs(scores1, scores2):
    from scipy.stats import entropy
    n_scores1 = len(scores1)
    n_scores2 = len(scores2)
    score_cost_matrix = np.zeros([n_scores1, n_scores2])
    for i in range(n_scores1):
        for j in range(n_scores2):
            score_cost_matrix[i, j] = entropy(scores1[i], scores2[j]) + entropy(scores2[j], scores1[i])
            #score_cost_matrix[i, j] = np.linalg.norm(scores1[i] - scores2[j])
            
    return score_cost_matrix 

def compute_euclide_cost(features1, features2):
    n_features1 = len(features1)
    n_features2 = len(features2)
    
    euclide_cost_matrix = np.zeros([n_features1, n_features2])
    for i in range(n_features1):
        for j in range(n_features2):
            euclide_cost_matrix[i, j] = np.linalg.norm(features1[i] - features2[j])
    
    return euclide_cost_matrix

def compute_filter_cost(features1, features2):
    pass
    

def compute_fc_costs(features1, features2, num_bins):
    n_features1 = len(features1)
    n_features2 = len(features2)
    features1 = features1.reshape(n_features1, features1.size() / n_features1)
    features2 = features2.reshape(n_features2, features2.size() / n_features2)
    
    features1_max = np.max(features1, axis=1)
    features2_max = np.max(features2, axis=1)
    
    features_max = features1_max + features2_max
    print np.max(features_max), np.median(features_max)
    
    cut_off = np.median(features_max)
    
    features1_histogram = np.zeros(n_features1, num_bins)
    features2_histogram = np.zeros(n_features2, num_bins)
    
    for i in range(n_features1):
        feature = features1[i, :]
        feature = feature[feature > 0]
        hist = np.histogram(feature, num_bins, [0, cut_off])
        features1_histogram[i, :] = hist / sum(hist)
    
    for i in range(n_features2):
        feature = features2[i, :]
        feature = feature[feature > 0]
        hist = np.histogram(feature, num_bins, [0, cut_off])
        features2_histogram[i, :] = hist / sum(hist)   
    
    feature_cost_matrix = np.zeros([n_features1, n_features2])
    for i in range(n_features1):
        for j in range(n_features2):
            h1 = features1_histogram[i, :]
            h2 = features2_histogram[j, :]
            feature_cost_matrix[i, j] = np.square(h1 - h2) / (h1 + h2)
            
    return feature_cost_matrix

def compute_feature_costs(feature1, feature2, feature_type):
    
    if feature_type == "label":
        costs = compute_label_costs(feature1, feature2)
    elif feature_type == "score":
        costs = compute_score_costs(feature1, feature2)
    elif feature_type in ["fc6", "fc7"]:
        costs = compute_fc_costs(feature1, feature2, 100)
    elif feature_type in ["fc8"]:
        costs = compute_euclide_cost(feature1, feature2)
    
    return costs
    
    
    

def tps_deep_basic_core(demofiles, demo_key1, demo_key2, deep_feature_types, deep_feature_weights_initial, deep_feature_weights_final, rough_init, init_tfm = None):
    print demo_key1, demo_key2
    demo1 = demofiles[demo_key1[0]][demo_key1[1]]
    demo2 = demofiles[demo_key2[0]][demo_key2[1]]
    demo_xyz1 = np.asarray(demo1["downsampled_cloud_xyz"])
    demo_xyz2 = np.asarray(demo2["downsampled_cloud_xyz"])
    demo_rgb1 = demo1['rgb']
    demo_rgb2 = demo2['rgb']
    
    query_features = {}
    for feature_type in deep_feature_types:
        learned_features = demo1["learned_features"]
        query_features[feature_type] = np.asarray(learned_features[feature_type])
        


    dataset_features = {}
    for feature_type in deep_feature_types:
        learned_features = demo2["learned_features"]
        dataset_features[feature_type] = np.asarray(learned_features[feature_type])
        
    if "label" in deep_feature_types:
        query_labels = query_features["label"]
        dataset_labels = dataset_features["label"]
    else:
        query_labels = np.asarray(demo1["learned_features"]["label"])
        dataset_labels = np.asarray(demo2["learned_features"]["label"])

        
        
        
    feature_costs = [compute_feature_costs(query_features[feature_type], dataset_features[feature_type], feature_type) for feature_type in deep_feature_types]         
        
    if init_tfm is not None:
        demo_xyz1 = demo_xyz1.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        demo_xyz2 = demo_xyz2.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
        
    result = registration_cost_and_tfm_and_corr_features(demo_xyz1, demo_xyz2, query_labels, dataset_labels, feature_costs, deep_feature_weights_initial, deep_feature_weights_final, rough_init)
    
    return result


def tps_deep_basic_sequential(demofiles, query_demo_keys, dataset_demo_keys, deep_feature_types, deep_feature_weights_initial, deep_feature_weights_final, rough_init, disable_label_consistent, init_tfm = None):

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
        
        
    query_features = {}
    for feature_type in deep_feature_types:
        feature = []
        for demo_key in query_demo_keys:
            learned_features = demofiles[demo_key[0]][demo_key[1]]["learned_features"]
            feature.append(np.asarray(learned_features[feature_type]))
        query_features[feature_type] = feature


    dataset_features = {}
    for feature_type in deep_feature_types:
        feature = []
        for demo_key in dataset_demo_keys:
            learned_features = demofiles[demo_key[0]][demo_key[1]]["learned_features"]
            feature.append(np.asarray(learned_features[feature_type]))
        dataset_features[feature_type] = feature
        
        
    if "label" in deep_feature_types:
        query_labels = query_features["label"]
        dataset_labels = dataset_features["label"]
    else:
        query_labels = []
        dataset_labels = []
        for demo_key in query_demo_keys:
            labels = demofiles[demo_key[0]][demo_key[1]]["learned_features"]["label"]
            query_labels.append(np.asarray(labels))
        for demo_key in dataset_demo_keys:
            labels = demofiles[demo_key[0]][demo_key[1]]["learned_features"]["label"]
            dataset_labels.append(np.asarray(labels))
    
    n_queries = len(query_demo_keys)
    n_datasets = len(dataset_demo_keys)
    valid_tasks = []
    for (i, query_key) in zip(range(n_queries), query_demo_keys):
        query_label = query_labels[i]
        query_valid_tasks = []
        for (j, dataset_key) in zip(range(n_datasets), dataset_demo_keys):
            dataset_label = dataset_labels[j]
            
            if disable_label_consistent:
                crossing_consistent = True
            else:
                # check whether one has crossing and the other does not
                crossing_consistent = ((3 in query_label) and (3 in dataset_label)) or (not (3 in query_label) and not (3 in dataset_label))
            
            if not crossing_consistent:
                print "filter out", query_key, dataset_key
            else:
                query_valid_tasks.append((i, j))      
        valid_tasks = valid_tasks + query_valid_tasks
        
        
        
        
    
    all_results = []    
    for (i, j) in valid_tasks:
        print query_demo_keys[i], dataset_demo_keys[j]
        result = registration_cost_and_tfm_and_corr_features(query_demo_xyzs[i], dataset_demo_xyzs[j], query_labels[i], dataset_labels[j], [compute_feature_costs(query_features[feature_type][i], dataset_features[feature_type][j], feature_type) for feature_type in deep_feature_types], deep_feature_weights_initial, deep_feature_weights_final, rough_init)
        all_results.append(result)


    
    
    query_results = {}
    num_dataset_demos = len(dataset_demo_keys)
    
    for query_key in query_demo_keys:
        costs = [np.inf] * num_dataset_demos
        res_costs = [np.inf] * num_dataset_demos
        fs = [None] * num_dataset_demos
        corrs = [None] * num_dataset_demos
                
        query_results[query_key] = {"costs": costs, "res_costs": res_costs, "fs": fs, "corrs": corrs}
        
    for (id, (i, j)) in zip(range(len(valid_tasks)), valid_tasks):
        query_key = query_demo_keys[i]
        res_cost, bend_cost, f, corr = all_results[id]
        query_results[query_key]["costs"][j] = bend_cost
        query_results[query_key]["res_costs"][j] = res_cost
        query_results[query_key]["fs"][j] = f
        query_results[query_key]["corrs"][j] = corr
        
    
    return {"results": query_results, "queries": query_demo_keys, "datasets": dataset_demo_keys, "feature_types": deep_feature_types}    
    
    
def tps_deep_basic_parallel(demofiles, query_demo_keys, dataset_demo_keys, deep_feature_types, deep_feature_weights_initial, deep_feature_weights_final, rough_init, disable_label_consistent, init_tfm = None):
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
        
        
    query_features = {}
    for feature_type in deep_feature_types:
        feature = []
        for demo_key in query_demo_keys:
            learned_features = demofiles[demo_key[0]][demo_key[1]]["learned_features"]
            feature.append(np.asarray(learned_features[feature_type]))
        query_features[feature_type] = feature


    dataset_features = {}
    for feature_type in deep_feature_types:
        feature = []
        for demo_key in dataset_demo_keys:
            learned_features = demofiles[demo_key[0]][demo_key[1]]["learned_features"]
            feature.append(np.asarray(learned_features[feature_type]))
        dataset_features[feature_type] = feature
        
        
    if "label" in deep_feature_types:
        query_labels = query_features["label"]
        dataset_labels = dataset_features["label"]
    else:
        query_labels = []
        dataset_labels = []
        for demo_key in query_demo_keys:
            labels = demofiles[demo_key[0]][demo_key[1]]["learned_features"]["label"]
            query_labels.append(np.asarray(labels))
        for demo_key in dataset_demo_keys:
            labels = demofiles[demo_key[0]][demo_key[1]]["learned_features"]["label"]
            dataset_labels.append(np.asarray(labels))
    
    n_queries = len(query_demo_keys)
    n_datasets = len(dataset_demo_keys)
    valid_tasks = []
    for (i, query_key) in zip(range(n_queries), query_demo_keys):
        query_label = query_labels[i]
        query_valid_tasks = []
        for (j, dataset_key) in zip(range(n_datasets), dataset_demo_keys):
            dataset_label = dataset_labels[j]
            
            # check whether one has crossing and the other does not
            
            if disable_label_consistent:
                crossing_consistent = True
            else:
                crossing_consistent = ((3 in query_label) and (3 in dataset_label)) or (not (3 in query_label) and not (3 in dataset_label))
                
            if not crossing_consistent:
                print "filter out", query_key, dataset_key
            else:
                query_valid_tasks.append((i, j))      
        valid_tasks = valid_tasks + query_valid_tasks
        
        
    
    all_results = Parallel(n_jobs=4, verbose=51)(delayed(registration_cost_and_tfm_and_corr_features)(query_demo_xyzs[i], dataset_demo_xyzs[j], query_labels[i], dataset_labels[j], [compute_feature_costs(query_features[feature_type][i], dataset_features[feature_type][j], feature_type) for feature_type in deep_feature_types], deep_feature_weights_initial, deep_feature_weights_final, rough_init)
                                                 for (i, j) in valid_tasks)

    
    query_results = {}
    num_dataset_demos = len(dataset_demo_keys)
    
    for query_key in query_demo_keys:
        costs = [np.inf] * num_dataset_demos
        res_costs = [np.inf] * num_dataset_demos
        fs = [None] * num_dataset_demos
        corrs = [None] * num_dataset_demos
                
        query_results[query_key] = {"costs": costs, "res_costs": res_costs, "fs": fs, "corrs": corrs}
        
    for (id, (i, j)) in zip(range(len(valid_tasks)), valid_tasks):
        query_key = query_demo_keys[i]
        res_cost, bend_cost, f, corr = all_results[id]
        query_results[query_key]["costs"][j] = bend_cost
        query_results[query_key]["res_costs"][j] = res_cost
        query_results[query_key]["fs"][j] = f
        query_results[query_key]["corrs"][j] = corr
        
    
    return {"results": query_results, "queries": query_demo_keys, "datasets": dataset_demo_keys, "feature_types": deep_feature_types}    



def tps_deep_basic_dataset(demofiles, query_demofiles, dataset_demofiles, net, deep_feature_types, deep_feature_weights_initial, deep_feature_weights_final, parallel, rough_init, disable_label_consistent, re_precompute = False, init_tfm = None):
    
    query_demo_keys= []
    dataset_demo_keys = []
    
    for demo_name in demofiles:
        if demo_name in query_demofiles:
            for seg_name in demofiles[demo_name]:
                query_demo_keys.append((demo_name, seg_name))
        if demo_name in dataset_demofiles:
            for seg_name in demofiles[demo_name]:
                dataset_demo_keys.append((demo_name, seg_name))

    if re_precompute:
        for demo_name in demofiles:
            for seg_name in demofiles[demo_name]:
                print "label", demo_name, seg_name
                
                demo = demofiles[demo_name][seg_name]
                demo_xyz = clouds.downsample(np.asarray(demo["cloud_xyz"]), DS_LEAF_SIZE)
                demo_rgb = demo['rgb']
                demo_label, demo_score, demo_features, valid_mask = predictCrossing3D(demo_xyz, demo_rgb, net, demo_name + "_" + seg_name)
                
                demo_xyz = demo_xyz[valid_mask]
                
                if "downsampled_cloud_xyz" in demo.keys():
                    demo["downsampled_cloud_xyz"][()] = demo_xyz
                else:
                    demo["downsampled_cloud_xyz"] = demo_xyz 
    
                
                if not "learned_features" in demo.keys():
                    demo.create_group("learned_features")
                    
                features_group = demo["learned_features"]
                for feature_name in demo_features:
                    if feature_name in features_group.keys():
                        features_group[feature_name][()] = demo_features[feature_name]
                    else:
                        features_group[feature_name] = demo_features[feature_name]           
                        
                if "label" in features_group.keys():
                    features_group["label"][()] = np.expand_dims(demo_label, axis=1) 
                else:
                    features_group["label"] = demo_label    
                    
                if "score" in features_group.keys():
                    features_group["score"][()] = demo_score      
                else:
                    features_group["score"] = demo_score     
                  
        demofiles.flush()

        
    if parallel:
        query_results = tps_deep_basic_parallel(demofiles, query_demo_keys, dataset_demo_keys, deep_feature_types, deep_feature_weights_initial, deep_feature_weights_final, rough_init, disable_label_consistent, init_tfm)
    else:
        query_results = tps_deep_basic_sequential(demofiles, query_demo_keys, dataset_demo_keys, deep_feature_types, deep_feature_weights_initial, deep_feature_weights_final, rough_init, disable_label_consistent, init_tfm)

    return query_results        
            

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--parallel", action="store_true")
parser.add_argument("--use_ar_init", action="store_true")
parser.add_argument("--tps_type", type=str, help="basic, segment, deep_basic")
parser.add_argument("--end_query", type=int, default=2)

parser.add_argument("--net_prototxt", help="File name for prototxt", type=str, default="")
parser.add_argument("--net_model", help="File name for learned model", type=str, default="")
parser.add_argument("--net_mean", help="File name for mean values", type=str, default="")
parser.add_argument("--re_precompute", help="Re-learn the features for each demo", action="store_true")
parser.add_argument("--deep_feature_types", type=str, default="", help="combination of fc6, fc7, fc8, lp1, lp2, pool1, pool2, pool5, score, label")
parser.add_argument("--deep_feature_weights_initial", type=str, default="")
parser.add_argument("--deep_feature_weights_final", type=str, default="")
parser.add_argument("--rough_init", action="store_true")
parser.add_argument("--disable_label_consistent", action="store_true")


args = parser.parse_args()


def main():
    task_dir = osp.join(demo_files_dir, args.demo_type)
    demo_h5_file = osp.join(task_dir, args.demo_type + ".h5")
    
    demofiles = h5py.File(demo_h5_file, "r+")    
    
    if args.tps_type in ["deep_basic"]:
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
            
        deep_feature_types = args.deep_feature_types
        deep_feature_types = deep_feature_types.replace(" ", "")
        if deep_feature_types == "":
            deep_feature_types = []
        else:
            deep_feature_types = deep_feature_types.split(',')
            
        if args.deep_feature_weights_initial == "":
            deep_feature_weights_initial = np.ones(len(deep_feature_types))
        else:
            deep_feature_weights_initial = args.deep_feature_weights_initial.split(',')
            deep_feature_weights_initial = [float(x) for x in deep_feature_weights_initial]
            if len(deep_feature_weights_initial) != len(deep_feature_types):
                raise Exception("deep_feature_weights_initial should be of the same length as features")
            
        if args.deep_feature_weights_final == "":
            deep_feature_weights_final = np.ones(len(deep_feature_types))
        else:
            deep_feature_weights_final = args.deep_feature_weights_final.split(',')
            deep_feature_weights_final = [float(x) for x in deep_feature_weights_final]
            if len(deep_feature_weights_final) != len(deep_feature_types):
                raise Exception("deep_feature_weights_final should be of the same length as features")
            
    print deep_feature_weights_initial
    print deep_feature_weights_final
        
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
        if demo_ind < args.end_query:
            query_demofiles.append(demo_name)
        
        dataset_demofiles.append(demo_name) 
            
        demo_ind += 1   

                            
    if args.tps_type == "basic":     
        results = tps_basic_dataset(demofiles, query_demofiles, dataset_demofiles, args.parallel, init_tfm)
    elif args.tps_type == "segment":
        results = tps_segment_dataset(demofiles, query_demofiles, dataset_demofiles, args.parallel, args.re_precompute, init_tfm)
    elif args.tps_type == "deep_basic":
        results = tps_deep_basic_dataset(demofiles, query_demofiles, dataset_demofiles, net, deep_feature_types, deep_feature_weights_initial, deep_feature_weights_final, args.parallel, args.rough_init, args.disable_label_consistent, args.re_precompute, init_tfm)

    
    if args.use_ar_init:
        use_ar = "use_ar"
    else:
        use_ar = "no_use_ar"
        
    if args.rough_init:
        use_rough_init = "use_rough"
    else:
        use_rough_init = "not_use_rough"
        
    if args.tps_type in["deep_basic"]:
        deep_feature_types_name = ""
        for feature_type in deep_feature_types:
            deep_feature_types_name = deep_feature_types_name + "_" + feature_type

        result_filename = osp.join(task_dir, "result_" + args.tps_type + "_" + use_ar + "_" + use_rough_init + "_" + str(args.end_query) + deep_feature_types_name + ".cp")
    else:
        result_filename = osp.join(task_dir, "result_" + args.tps_type + "_" + use_ar + "_" + use_rough_init + "_" + str(args.end_query) + ".cp")
    f = open(result_filename, 'wb')
    pickle.dump(results, f)
    
    
            
if __name__ == "__main__":
    main()

            
            
            
            
            
            
            


