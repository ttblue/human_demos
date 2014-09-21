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

def tps_segment_data(demofiles, demo_key, init_tfm = None):
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


def tps_segment(demofiles, demo_key1, demo_key2, parallel, init_tfm = None):
    tps_segment_crossing_infos1 = tps_segment_data(demofiles, demo_key1, init_tfm)
    tps_segment_crossing_infos2 = tps_segment_data(demofiles, demo_key2, init_tfm)
    
    if parallel:
        results = Parallel(n_jobs=4, verbose=51)(delayed(tps_segment_registration)(info1[0:5], info2[0:5], info1[6], info2[6], np.eye(4), x_weights=info1[5], reg=0.01) 
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
    
    print min(costs)
    
    return fs[choice_ind], corrs[choice_ind], costs[choice_ind]
    

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--parallel", action="store_true")
parser.add_argument("--use_ar_init", action="store_true")

args = parser.parse_args()


def main():
    task_dir = osp.join(demo_files_dir, args.demo_type)
    demo_h5_file = osp.join(task_dir, args.demo_type + ".h5")
    
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
    
    
    demo_keys = []
    
    for demo_name in demofiles:
        for seg_name in demofiles[demo_name]:
            demo_keys.append((demo_name, seg_name))
            
    num_demos = len(demo_keys)
    
    results = {}
    
    for i in range(len(demo_keys)):
        for j in range(len(demo_keys)):
            if i != j:
                demo_key1 = demo_keys[i]
                demo_key2 = demo_keys[j]
                
                cost, f, corr = tps_segment(demofiles, demo_key1, demo_key2, args.parallel)
                
                results[(i, j)] = (cost, f, corr)
            else:
                results[(i, j)] = (0, None, None)

            
if __name__ == "__main__":
    main()

            
            
            
            
            
            
            


