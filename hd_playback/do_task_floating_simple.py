#!/usr/bin/env python
from __future__ import division
import argparse
usage="""

Run in simulation with a translation and a rotation of fake data:
python do_task.py --demo_type=overhand --fake_data_demo=demo00001 --fake_data_segment=seg00 --execution=0 --animation=1 --select_manual --fake_data_transform .1 .1 .1 .1 .1 .1


Run in simulation choosing the closest demo, single threaded
./do_task.py demo_full/demo_full.h5 --fake_data_demo=demo_full_0 --fake_data_segment=seg00 --execution=0 --animation=1 --parallel=0

Actually run on the robot without pausing or animating
./do_task.py demo_full/demo_full.h5 --execution=1 --animation=0


"""
parser = argparse.ArgumentParser(usage=usage)


parser.add_argument("--demo_type", type=str)
parser.add_argument("--use_diff_length", action="store_true", default=False)


parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--downsample", help="downsample traj.", type=int, default=1)

parser.add_argument("--prompt", action="store_true")
parser.add_argument("--show_neighbors", action="store_true")
parser.add_argument("--select", default="manual")
parser.add_argument("--log", action="store_true")

parser.add_argument("--fake_data_demo", type=str)
parser.add_argument("--fake_data_segment",type=str)
parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
    default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")
parser.add_argument("--fake_rope", type=str)
parser.add_argument("--choose_demo", type=str)
parser.add_argument("--choose_seg", type=str)
parser.add_argument("--init_perturb", type=str)


parser.add_argument("--pot_threshold",type=float, default=15)

parser.add_argument("--use_ar_init", action="store_true", default=False)
parser.add_argument("--use_base", action="store_true", default=False)
parser.add_argument("--not_allow_base", help="dont allow base movement when use_base", action="store_true", default=False)
parser.add_argument("--early_stop_portion", help="stop early in the final segment to avoid bullet simulation problem", type=float, default=0.5)
parser.add_argument("--no_traj_resample", action="store_true", default=False)

parser.add_argument("--interactive",action="store_true", default=False)
parser.add_argument("--remove_table", action="store_true")
parser.add_argument("--step", type=int, default=3)
parser.add_argument("--no_display",action="store_true", default=False)


parser.add_argument("--friction", help="friction value in bullet", type=float, default=1.0)

parser.add_argument("--max_steps_before_failure", type=int, default=-1)
parser.add_argument("--tps_bend_cost_init", type=float, default=10)#1)
parser.add_argument("--tps_bend_cost_final", type=float, default=.01) #.001
parser.add_argument("--tps_bend_cost_final_search", type=float, default=.001) # .0001 #.00001
parser.add_argument("--tps_n_iter", type=int, default=50)

parser.add_argument("--closest_rope_hack", action="store_true", default=False)
parser.add_argument("--closest_rope_hack_thresh", type=float, default=0.01)
parser.add_argument("--cloud_downsample", type=float, default=.01)

parser.add_argument("--use_crossings", action="store_true", default=False)
parser.add_argument("--use_rotation", action="store_true", default=False)
parser.add_argument("--use_crits", action="store_true", default=False)
parser.add_argument("--test_success", action="store_true", default=False)
parser.add_argument("--force_points", action="store_true", default=False)



args = parser.parse_args()


"""
Workflow:
1. Fake data + animation only
--fake_data_segment=xxx --execution=0
2. Fake data + Gazebo. Set Gazebo to initial state of fake data segment so we'll execute the same thing.
--fake_data_segment=xxx --execution=1
This is just so we know the robot won't do something stupid that we didn't catch with openrave only mode.
3. Real data + Gazebo
--execution=1
The problem is that the gazebo robot is in a different state from the real robot, in particular, the head tilt angle. TODO: write a script that sets gazebo head to real robot head
4. Real data + Real execution.
--execution=1

The question is, do you update the robot's head transform.
If you're using fake data, don't update it.

"""
import os, h5py, time, os.path as osp
import cPickle, pickle
import numpy as np, numpy.linalg as nlg
import math

import scipy
from scipy.spatial import *

import openravepy, trajoptpy

from hd_rapprentice import registration, animate_traj, \
     plotting_openrave, task_execution, resampling, \
     ropesim_floating, rope_initialization
from hd_rapprentice.registration import ThinPlateSpline, Affine, Composition
from hd_utils import clouds, math_utils as mu, cloud_proc_funcs
#from hd_utils.pr2_utils import get_kinect_transform
from hd_utils.colorize import *
from hd_utils.utils import avg_transform
from hd_utils import transformations
from hd_utils.defaults import demo_files_dir, hd_data_dir,\
        ar_init_dir, ar_init_demo_name, ar_init_playback_name, \
        tfm_head_dof, tfm_bf_head, cad_files_dir, init_state_perturbs_dir
from knot_classifier import calculateCrossings, calculateMdp, calculateMdp2, isKnot, remove_crossing, pairs_to_dict



L_POSTURES = {'side': np.array([[-0.98108876, -0.1846131 ,  0.0581623 ,  0.10118172],
                                [-0.19076337,  0.97311662, -0.12904799,  0.68224057],
                                [-0.03277475, -0.13770277, -0.98993119,  0.91652485],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]]) }

R_POSTURES = {'side' : np.array([[-0.98108876,  0.1846131 ,  0.0581623 ,  0.10118172],
                                 [ 0.19076337,  0.97311662,  0.12904799, -0.68224057],
                                 [-0.03277475,  0.13770277, -0.98993119,  0.91652485],
                                 [ 0.        ,  0.        ,  0.        ,  1.        ]]) }

DIST_ANGS = {1.594436429147036e-16: 0.0,
 0.003844679890991385: 0.021052631578947368,
 0.0076832681820952382: 0.042105263157894736,
 0.011514063622777539: 0.063157894736842107,
 0.015335368416271792: 0.084210526315789472,
 0.01914548897203729: 0.10526315789473684,
 0.022942736656352665: 0.12631578947368421,
 0.026725428540714153: 0.14736842105263157,
 0.030491888147703532: 0.16842105263157894,
 0.03424044619399793: 0.18947368421052632,
 0.037969441330191346: 0.21052631578947367,
 0.041677220877099325: 0.23157894736842105,
 0.04536214155822138: 0.25263157894736843,
 0.049022570228036196: 0.27368421052631581,
 0.052656884595806502: 0.29473684210526313,
 0.056263473944573449: 0.31578947368421051,
 0.059840739845021411: 0.33684210526315789,
 0.063387096863896861: 0.35789473684210527,
 0.066900973266667774: 0.37894736842105264,
 0.070380811714111355: 0.40000000000000002}


INTERACTIVE_FLAG = False

class Globals:
    env = None
    sim = None
    viewer = None

def move_sim_arms_to_side():
    """Moves the simulated arms to the side."""
    Globals.sim.grippers['r'].set_toolframe_transform(R_POSTURES['side'])
    Globals.sim.grippers['l'].set_toolframe_transform(L_POSTURES['side'])

DS_SIZE = .025

def smaller_ang(x):
    return (x + np.pi)%(2*np.pi) - np.pi

def closer_ang(x,a,dr=0):
    """
    find angle y (==x mod 2*pi) that is close to a
    dir == 0: minimize absolute value of difference
    dir == 1: y > x
    dir == 2: y < x
    """
    if dr == 0:
        return a + smaller_ang(x-a)
    elif dr == 1:
        return a + (x-a)%(2*np.pi)
    elif dr == -1:
        return a + (x-a)%(2*np.pi) - 2*np.pi

def closer_angs(x_array,a_array,dr=0):

    return [closer_ang(x, a, dr) for (x, a) in zip(x_array, a_array)]


def split_trajectory_by_gripper(seg_info, pot_angle_threshold, ms_thresh=2):
    lgrip = np.asarray(seg_info["l"]["pot_angles"])
    rgrip = np.asarray(seg_info["r"]["pot_angles"])

    #print rgrip
    #print lgrip

    thresh = pot_angle_threshold # open/close threshold

    # TODO: Check this.
    n_steps = len(lgrip) - 1

    # indices BEFORE transition occurs
    l_openings = np.flatnonzero((lgrip[1:] >= thresh) & (lgrip[:-1] < thresh))
    r_openings = np.flatnonzero((rgrip[1:] >= thresh) & (rgrip[:-1] < thresh))
    l_closings = np.flatnonzero((lgrip[1:] < thresh) & (lgrip[:-1] >= thresh))
    r_closings = np.flatnonzero((rgrip[1:] < thresh) & (rgrip[:-1] >= thresh))

    before_transitions = np.r_[l_openings, r_openings, l_closings, r_closings]
    after_transitions = before_transitions+1
    seg_starts = np.unique(np.r_[0, after_transitions])
    seg_ends = np.unique(np.r_[before_transitions, n_steps-1])

    lr_open = {lr:[] for lr in 'lr'}
    new_seg_starts = []
    new_seg_ends = []
    for i in range(len(seg_starts)):
        if seg_ends[i]- seg_starts[i] >= ms_thresh:
            new_seg_starts.append(seg_starts[i])
            new_seg_ends.append(seg_ends[i])
            lval = True if lgrip[seg_starts[i]] >= thresh else False
            lr_open['l'].append(lval)
            rval = True if rgrip[seg_starts[i]] >= thresh else False
            lr_open['r'].append(rval)


    return new_seg_starts, new_seg_ends, lr_open


def fuzz_cloud(xyz, seg_group=None, init_tfm=None):
    #return xyz, (len(xyz),0,0)
    cross_section = observe_cloud(xyz, radius=0.005, upsample_rad=4)
    return np.vstack([xyz, cross_section]), (len(xyz),len(cross_section),0)
    table_pts = get_table_grid(xyz, 10, seg_group, init_tfm)
    #table_pts = get_table_fuzz(observe_cloud(xyz, radius = 0.025, upsample_rad=2), 100, seg_group, init_tfm)
    #xyz[:,2] = Globals.table_height + 0.1
    return np.vstack([xyz, cross_section, table_pts]), (len(xyz), len(cross_section), len(table_pts))


def observe_cloud(pts=None, radius=0.005, upsample=0, upsample_rad=1):
    """
    If upsample > 0, the number of points along the rope's backbone is resampled to be upsample points
    If upsample_rad > 1, the number of points perpendicular to the backbone points is resampled to be upsample_rad points, around the rope's cross-section
    The total number of points is then: (upsample if upsample > 0 else len(self.rope.GetControlPoints())) * upsample_rad

    Move to ropesim and/or ropesim_floating?
    """
    if pts == None: pts = Globals.sim.rope.GetControlPoints()
    half_heights = Globals.sim.rope.GetHalfHeights()
    if upsample > 0:
        lengths = np.r_[0, half_heights * 2]
        summed_lengths = np.cumsum(lengths)
        assert len(lengths) == len(pts)
        pts = math_utils.interp2d(np.linspace(0, summed_lengths[-1], upsample), summed_lengths, pts)
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


def get_table_fuzz(xyz, length=50, seg_group=None, init_tfm=None):
    if seg_group == None:
        flat_points = xyz[:,:-1]
    else:
        orig_space_xyz = (xyz - init_tfm[:3,3][None,:]).dot(np.linalg.pinv(init_tfm[:3,:3]).T)
        flat_points = np.array([clouds.XYZ_to_xy(*pt) for pt in orig_space_xyz])
    if seg_group == None:
        return np.hstack([flat_points, np.ones((len(flat_points),1))*(Globals.table_height+0.1)])
    else:
        depth_xyz = clouds.depth_to_xyz(seg_group["depth"][:])
        depth_projection = np.array([depth_xyz[int(round(pt[1])), int(round(pt[0]))] for pt in flat_points])
        ret = depth_projection #flatten_cloud(depth_projection)
        return np.array(ret).dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]


def get_table_grid(xyz, size=25, seg_group=None, init_tfm=None):
    if seg_group == None:
        flat_points = xyz[:,:-1]
    else:
        orig_space_xyz = (xyz - init_tfm[:3,3][None,:]).dot(np.linalg.pinv(init_tfm[:3,:3]).T)
        flat_points = np.array([clouds.XYZ_to_xy(*pt) for pt in orig_space_xyz])
    x1 = np.max(flat_points[:,0])*1.01; x2 = np.min(flat_points[:,0])*.99
    y1 = np.max(flat_points[:,1])*1.01; y2 = np.min(flat_points[:,1])*.99
    grid = np.vstack([np.array([[i,j,0] for i in np.linspace(x1,x2,size)]) for j in np.linspace(y1,y2,size)])
    if seg_group == None:
        grid[:,2] = Globals.table_height+0.1
        return grid
    else:
        depth_xyz = clouds.depth_to_xyz(seg_group["depth"][:])
        depth_projection = np.array([depth_xyz[int(round(pt[1])), int(round(pt[0]))] for pt in grid])
        ret = flatten_cloud(depth_projection)
        return np.array(ret).dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]


def get_voronoi(xyz, seg_group=None, init_tfm=None):
    if seg_group == None:
        flat_points = xyz[:,:-1]
    else:
        orig_space_xyz = (xyz - init_tfm[:3,3][None,:]).dot(np.linalg.pinv(init_tfm[:3,:3]).T)
        flat_points = np.array([clouds.XYZ_to_xy(*pt) for pt in orig_space_xyz])
    x1 = np.max(flat_points[:,0])+.01; x2 = np.min(flat_points[:,0])-.01
    y1 = np.max(flat_points[:,1])+.01; y2 = np.min(flat_points[:,1])-.01
    vor = Voronoi(flat_points)
    vertices = vor.vertices
    vertices = vertices[(vertices[:,0] < x1) & (vertices[:,0] > x2) & (vertices[:,1] < y1) & (vertices[:,1] > y2)]
    if seg_group == None:
        vertices = np.hstack([vertices, np.ones((len(vertices),1))*(Globals.table_height+0.1)])
        return vertices
    else:
        depth_xyz = clouds.depth_to_xyz(seg_group["depth"][:])
        depth_projection = np.array([depth_xyz[int(round(pt[1])), int(round(pt[0]))] for pt in vertices])
        ret = flatten_cloud(depth_projection)
        return np.array(ret).dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]


def flatten_cloud(xyz):
    U,S,Vt = np.linalg.svd(np.cov(xyz.T))
    n = U[:,2]/np.linalg.norm(U[:,2])    #direction of least variance; normal to the best-fit plane
    new_xyz = np.array([pt - pt.dot(n)*n for pt in xyz])  #project points onto best-fit plane
    new_xyz = new_xyz - np.mean(new_xyz, axis=0) + np.mean(xyz, axis=0) #recenter
    return new_xyz


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


def reflect_across_median(xyz, axis):
    """
    rotates xyz by theta around the median along the x, y dimensions
    """
    median = np.median(xyz, axis=0)
    centered_xyz = xyz - median
    centered_xyz[:,axis] = -1*centered_xyz[:,axis]
    new_xyz = centered_xyz + median
    return new_xyz


def rotate_by_pca(xyz0, xyz1):
    """
    rotates xyz0 to align its principal component axes with those of xyz1
    """
    pca0,_,_ = np.linalg.svd(np.cov(xyz0.T))
    pca1,_,_ = np.linalg.svd(np.cov(xyz1.T))
    median = np.median(xyz0, axis=0)
    centered_xyz = xyz0 - median
    aligned_xyz = centered_xyz.dot(pca0) # aligned to global axes
    aligned_xyz = aligned_xyz - np.median(aligned_xyz) #recenter to avoid numerical issues - hacky
    aligned_xyz = aligned_xyz.dot(np.linalg.inv(pca1)) # aligned to PC axes of xyz1
    aligned_xyz = aligned_xyz - np.median(aligned_xyz) #recenter to avoid numerical issues -hacky
    new_xyz = aligned_xyz + median
    return new_xyz


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


def get_crossings(rope_points):
    crossings_pattern, _, points = calculateCrossings(rope_points, get_points=True)
    crossings_locations = np.array(points) #np.array([rope_points[i] for i in np.sort(np.array(list(cross_pairs)).flatten())])
    return crossings_pattern, crossings_locations


def get_critical_points_sim(rope_points):
    sim_xyzc = get_labeled_rope_sim(rope_points)
    if sim_xyzc[-1][-1] != 0:
        print "remove last sim crossing (from end)"
        sim_xyzc = remove_crossing(sim_xyzc,-1)
    elif sim_xyzc[0][-1] != 0:
        print "remove first sim crossing (from end)"
        sim_xyzc = remove_crossing(sim_xyzc,0)
    crit_pts = [pt[:-1] for pt in sim_xyzc if pt[-1]!=0]
    return np.array(crit_pts).reshape(len(crit_pts),3)


def get_critical_points_demo(seg_group):
    demo_xyzc = get_labeled_rope_demo(seg_group)
    if demo_xyzc[-1][-1] != 0: #last point of demo is a crossing
        print "remove last demo crossing (from end)"
        demo_xyzc = remove_crossing(demo_xyzc,-1)
    elif demo_xyzc[0][-1] != 0: #first point of demo is a crossing
        print "remove first demo crossing (from end)"
        demo_xyzc = remove_crossing(demo_xyzc,0)
    crit_pts = [pt[:-1] for pt in demo_xyzc if pt[-1]!=0]
    return np.array(crit_pts).reshape(len(crit_pts),3)


def get_labeled_rope_sim(rope_points, get_pattern=False):
    pattern, _, inds = calculateCrossings(rope_points, get_inds=True)
    crossing_ind = 0
    labeled_rope = np.zeros((len(rope_points),4))
    for i in range(len(rope_points)):
        labeled_rope[i,:3] = rope_points[i]
        if i in inds:
            labeled_rope[i,3] = pattern[crossing_ind]
            crossing_ind+=1
    if not get_pattern:
        return labeled_rope
    else:
        if labeled_rope[-1][-1] != 0:
            print "remove last sim crossing (from end)"
            labeled_rope = remove_crossing(labeled_rope,-1)
        elif labeled_rope[0][-1] != 0:
            print "remove first sim crossing (from end)"
            labeled_rope = remove_crossing(labeled_rope,0)
        pattern = [pt[-1] for pt in labeled_rope if pt[-1]!=0]
        return labeled_rope, pattern


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


def get_pattern_sim(rope_points):
    sim_xyzc = get_labeled_rope_sim(rope_points)
    if sim_xyzc[-1][-1] != 0:
        print "remove last sim crossing (from end)"
        sim_xyzc = remove_crossing(sim_xyzc,-1)
    elif sim_xyzc[0][-1] != 0:
        print "remove first sim crossing (from end)"
        sim_xyzc = remove_crossing(sim_xyzc,0)
    sim_pattern = [pt[-1] for pt in sim_xyzc if pt[-1]!=0]
    return sim_pattern


def get_pattern_demo(seg_group):
    demo_xyzc = get_labeled_rope_demo(seg_group)

    if demo_xyzc[-1][-1] != 0: #last point of demo is a crossing
        print "remove last demo crossing (from end)"
        demo_xyzc = remove_crossing(demo_xyzc,-1)
    elif demo_xyzc[0][-1] != 0: #first point of demo is a crossing
        print "remove first demo crossing (from end)"
        demo_xyzc = remove_crossing(demo_xyzc,0)
    demo_pattern = [pt[-1] for pt in demo_xyzc if pt[-1]!=0]
    return demo_pattern


def pickle_dump(xyz,filename):
    file1 = open(filename, 'wb')
    xyz = pickle.dump(xyz, file1)


def pickle_load(filename):
    file1 = open(filename, 'rb')
    xyz = pickle.load(file1)
    return xyz


FEASIBLE_REGION = [[.3, -.5], [.8, .5]]# bounds on region robot can hope to tie rope in

def place_in_feasible_region(xyz):
    max_xyz = np.max(xyz, axis=0)
    min_xyz = np.min(xyz, axis=0)
    offset = np.zeros(3)
    for i in range(2):
        if min_xyz[i] < FEASIBLE_REGION[0][i]:
            offset[i] = FEASIBLE_REGION[0][i] - min_xyz[i]
        elif max_xyz[i] > FEASIBLE_REGION[1][i]:
            offset[i] = FEASIBLE_REGION[1][i] - max_xyz[i]
    return xyz + offset

def binarize_gripper(angle, pot_angle_threshold):
    open_angle = .08
    closed_angle = 0
    if angle >= pot_angle_threshold: return open_angle
    else: return closed_angle


def set_gripper_sim(lr, is_open, prev_is_open):
    mult = 5
    open_angle = .08 * mult
    closed_angle = .02 * mult

    target_val = open_angle if is_open else closed_angle

    if is_open and not prev_is_open:
        Globals.sim.release_rope(lr)

    # execute gripper open/close trajectory
    start_val = Globals.sim.grippers[lr].get_gripper_joint_value()
    joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .02))
    for val in joint_traj:
        Globals.sim.grippers[lr].set_gripper_joint_value(val)
        dist = get_finger_dist(lr)
        # try:
        #     DIST_ANGS[dist] = val
        # except AttributeError as atterr:
        #     DIST_ANGS = {dist: val}
        Globals.sim.step()
        if Globals.viewer:
            Globals.viewer.Step()
            if args.interactive or INTERACTIVE_FLAG: Globals.viewer.Idle()
    # add constraints if necessary
    if not is_open and prev_is_open:
        if not Globals.sim.grab_rope(lr):
            return False
    return True

def unwrap_arm_traj_in_place(traj):
    assert traj.shape[1] == 7
    for i in [2,4,6]:
        traj[:,i] = np.unwrap(traj[:,i])
    return traj

def unwrap_in_place(t):
    # TODO: do something smarter than just checking shape[1]
    if t.shape[1] == 7:
        unwrap_arm_traj_in_place(t)
    elif t.shape[1] == 14:
        unwrap_arm_traj_in_place(t[:,:7])
        unwrap_arm_traj_in_place(t[:,7:])
    else:
        raise NotImplementedError

def exec_traj_sim(lr_traj, animate=True, ljoints=None, rjoints=None):
    def sim_callback(i):
        Globals.sim.step()
    if args.no_display:
        animate = False

    lhmats_up, rhmats_up = ropesim_floating.retime_hmats(lr_traj['l'], lr_traj['r'])
    
    if ljoints != None:
        ljoints = resample_joint_angs(ljoints, len(lhmats_up))
        rjoints = resample_joint_angs(rjoints, len(rhmats_up))

    # in simulation mode, we must make sure to gradually move to the new starting position
    curr_rtf  = Globals.sim.grippers['r'].get_toolframe_transform()
    curr_ltf  = Globals.sim.grippers['l'].get_toolframe_transform()

    l_transition_hmats, r_transition_hmats = ropesim_floating.retime_hmats([curr_ltf, lhmats_up[0]], [curr_rtf, rhmats_up[0]])

    animate_traj.animate_floating_traj(l_transition_hmats, r_transition_hmats,
                                       Globals.sim, pause=False,
                                       callback=sim_callback, step_viewer=animate, step=args.step)
    animate_traj.animate_floating_traj_angs(lhmats_up, rhmats_up, ljoints, rjoints, Globals.sim, pause=False,
                                       callback=sim_callback, step_viewer=animate, step=args.step)
    return True

def get_finger_trajs(lr, traj, is_open=False):
    robot = Globals.sim.grippers[lr].robot
    old_val = Globals.sim.grippers[lr].get_gripper_joint_value()
    if is_open:
        Globals.sim.grippers[lr].set_gripper_joint_value(.4)
        print "IS OPEN"
    elif old_val > 0.2:
        print "gripper is already open"
    tt_tfm = Globals.sim.grippers[lr].get_toolframe_transform() # == robot.GetLink(lr+"_gripper_tool_frame").GetTransform()

    ltraj, rtraj = np.zeros((len(traj), 4, 4)), np.zeros((len(traj), 4, 4))
    for i in range(len(traj)):
        tt2ltip, tt2rtip = Globals.sim.grippers[lr].get_tt2ftips(Globals.sim.grippers[lr].get_gripper_joint_value())
        ltraj[i], rtraj[i] = traj[i].dot(tt2ltip), traj[i].dot(tt2rtip)
    return ltraj, rtraj


def tooltip_from_fingers(lr, prev, lhmat, rhmat, joint_val):
    robot = Globals.sim.grippers[lr].robot

    tt_tfm = prev

    tt2ltip, tt2rtip = Globals.sim.grippers[lr].get_tt2ftips(joint_val)
    pts = np.vstack([[0.015,0,0,1],[-0.015,0,0,1], [0,0,0,1]]).T
    pts1 = np.vstack([tt_tfm.dot(tt2ltip).dot(pts)[:-1].T, tt_tfm.dot(tt2rtip).dot(pts)[:-1].T])
    pts2 = np.vstack([lhmat.dot(pts)[:-1].T, rhmat.dot(pts)[:-1].T])
    c1 = pts1 - pts1.mean(0)
    c2 = pts2 - pts2.mean(0)
    H = c1.T.dot(c2)
    U,S,Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    if np.linalg.det(R) < 0: 
        Vt[2,:] *= -1
        R = Vt.T.dot(U.T)
    rot_tfm = np.eye(4)
    rot_tfm[:3,:3] = R
    rot_tfm[:3,3] = -R.dot(pts1.mean(0)) + pts2.mean(0)
    new_tfm = rot_tfm.dot(tt_tfm)
    return new_tfm


def joint_angles_from_fingers(ltraj, rtraj, is_open=False):
    if is_open:
        min_val = 0.15
        #return [0.4 for i in range(len(ltraj))] #tkl
    else:
        #min_val = 0.1
        return [0.08 for i in range(len(ltraj))]
    if len(ltraj) != len(rtraj):
        print "finger trajectories are of different lengths"
        return
    joint_angles = []
    for i in range(len(ltraj)):
        dist = np.linalg.norm(ltraj[i,:3,3]-rtraj[i,:3,3])
        closest_dist = DIST_ANGS.keys()[np.argmin([abs(d-dist) for d in DIST_ANGS.keys()])]
        joint_angles.append(max(DIST_ANGS[closest_dist], min_val))
        #print dist, closest_dist, DIST_ANGS[closest_dist]
    return joint_angles


def get_finger_dist(lr):
    robot = Globals.sim.grippers[lr].robot
    ltip_tfm = np.eye(4); ltip_tfm[:3,3] = Globals.sim.grippers[lr].link2finger['l']
    rtip_tfm = np.eye(4); rtip_tfm[:3,3] = Globals.sim.grippers[lr].link2finger['r']
    l_finger_tfm = robot.GetLink("l_gripper_l_finger_tip_link").GetTransform().dot(ltip_tfm)
    r_finger_tfm = robot.GetLink("l_gripper_r_finger_tip_link").GetTransform().dot(rtip_tfm)
    dist = np.linalg.norm(l_finger_tfm[:3,3]-r_finger_tfm[:3,3])
    return dist

def resample_joint_angs(joints, new_len):
    x = range(new_len)
    xp = range(len(joints))
    fp = joints
    y = np.interp(x,xp,fp)
    return y

def find_closest_manual(demofiles):
    """for now, just prompt the user"""

    if not isinstance(demofiles, list):
        demofiles = [demofiles]

    print "choose from the following options (type an integer)"
    seg_num = 0
    demotype_num = 0
    keys = {}
    is_finalsegs = {}
    for demofile in demofiles:
        print "Type %i: "%(demotype_num + 1)
        for demo_name in demofile:
            if demo_name != "ar_demo":
                if 'done' in demofile[demo_name].keys():
                    final_seg_id = len(demofile[demo_name].keys()) - 2
                else:
                    final_seg_id = len(demofile[demo_name].keys()) - 1


                for seg_name in demofile[demo_name]:
                    if seg_name != 'done':
                        keys[seg_num] = (demotype_num, demo_name, seg_name)
                        print "%i: %i, %s, %s"%(seg_num, demotype_num+1, demo_name, seg_name)

                        if seg_name == "seg%02d"%(final_seg_id):
                            is_finalsegs[seg_num] = True
                        else:
                            is_finalsegs[seg_num] = False

                        seg_num += 1
        demotype_num +=1
    choice_ind = task_execution.request_int_in_range(seg_num)
    if demotype_num == 1:
        return (keys[choice_ind][1],keys[choice_ind][2]) , is_finalsegs[choice_ind]
    else:
        return keys[choice_ind], is_finalsegs[choice_ind]


def registration_cost(xyz0, xyz1, num_iters=30, block_lengths=None):
    scaled_xyz0, _ = registration.unit_boxify(xyz0)
    scaled_xyz1, _ = registration.unit_boxify(xyz1)
    f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, reg_init=args.tps_bend_cost_init, reg_final = args.tps_bend_cost_final_search, 
                                    rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=num_iters, block_lengths=block_lengths)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    return cost


def registration_cost_and_tfm(xyz0, xyz1, num_iters=30, block_lengths=None):
    scaled_xyz0, src_params = registration.unit_boxify(xyz0)
    scaled_xyz1, targ_params = registration.unit_boxify(xyz1)
    f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, reg_init=args.tps_bend_cost_init, reg_final = args.tps_bend_cost_final_search, 
            rad_init = .1, rad_final = .0005, rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=30, plotting=True, block_lengths=block_lengths, Globals=Globals)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    g = registration.unscale_tps_3d(g, targ_params, src_params)
    f = registration.unscale_tps_3d(f, src_params, targ_params)
    return (cost, f, g)


def registration_cost_with_rotation(xyz0, xyz1, block_lengths=None, known_theta=None):
    #rad_angs = np.linspace(-np.pi+np.pi*(float(10)/180), np.pi,36)
    #rad_angs = np.linspace(-np.pi+np.pi*(float(45)/180), np.pi,8)
    if known_theta:
        rad_angs = [known_theta]
    else:
        rad_angs = np.linspace(-np.pi+np.pi*(float(30)/180), np.pi,12)
    costs = np.zeros(len(rad_angs))
    for i in range(len(rad_angs)):
        rotated_demo = rotate_about_median(xyz0, rad_angs[i])
        costs[i] = registration_cost(rotated_demo, xyz1, 3, block_lengths=block_lengths)
    theta = rad_angs[np.argmin(costs)]
    rotated_demo = rotate_about_median(xyz0, theta)
    cost, f, g = registration_cost_and_tfm(rotated_demo, xyz1, block_lengths=block_lengths)

    rotate, unrotate = rotations_from_ang(theta,np.median(xyz0, axis=0))
    f = Composition([rotate, f])
    g = Composition([g, unrotate])

    return (cost, f, g, theta)


def registration_cost_with_pca_rotation(xyz0, xyz1, critical_points=0):
    costs = []; tfms = []
    pca_demo = rotate_by_pca(xyz0, xyz1)
    ptclouds = [pca_demo, reflect_across_median(pca_demo, 1), rotate_about_median(pca_demo, np.pi)]
    for ptcloud in ptclouds:
        costs.append(registration_cost(ptcloud, xyz1, 3, critical_points))
    ptcloud = ptclouds[np.argmin(costs)]
    cost,f,g = registration_cost_and_tfm(ptcloud, xyz1, critical_points=critical_points)
    pca0,_,_ = np.linalg.svd(np.cov(xyz0.T)); pca1,_,_ = np.linalg.svd(np.cov(xyz1.T))
    rmat = pca0.dot(np.linalg.inv(pca1))
    if np.argmin(costs) == 2:
        rotation_matrix = np.eye(3)
        rotation_matrix[0:2, 0:2] = np.array([[-1, 0],[0, -1]])
        rmat = rmat.dot(rotation_matrix)
    shift_from_origin = Affine(np.eye(3), np.median(xyz0, axis=0))
    rotate = Affine(rmat, np.zeros((3,)))
    unrotate = Affine(np.linalg.pinv(rmat), np.zeros((3,)))
    shift_to_origin = Affine(np.eye(3), -1*np.median(xyz0, axis=0))

    if np.argmin(costs) == 1:
        flip_mat = np.eye(3)
        flip_mat[:,1] = -1*flip_mat[:,1]
        flip_tfm = Affine(flip_mat, np.zeros((3,)))
        f = Composition([shift_to_origin, rotate, flip_tfm, shift_from_origin, f])
        g = Composition([g, shift_to_origin, flip_tfm, unrotate, shift_from_origin])
    else:
        f = Composition([shift_to_origin, rotate, shift_from_origin, f])
        g = Composition([g, shift_to_origin, unrotate, shift_from_origin])

    return (np.min(costs), f, g, rmat)


def registration_cost_crit(xyz0, xyz1, crit_demo, crit_sim, block_lengths=None, known_theta=None):
    if block_lengths == None: block_lengths = [(len(xyz0), len(xyz1))]
    crit_num = block_lengths[0][0]
    # if crit_demo == None or len(crit_demo)==0:
    #     cost1, f1, g1, theta1, = registration_cost_with_rotation(xyz0, xyz1, block_lengths=block_lengths, known_theta=known_theta)
    #     xyz0 = np.vstack([xyz0[:crit_num][::-1], xyz0[crit_num:][::-1]])
    #     cost2, f2, g2, theta2 = registration_cost_with_rotation(xyz0[::-1], xyz1, block_lengths=block_lengths, known_theta=known_theta)
    #     if cost2 < cost1:
    #         return cost2, f2, g2, theta2
    #     else:
    #         return cost1, f1, g1, theta1
    if not args.force_points:
        if len(crit_sim) == len(crit_demo):
            pattern, _, inds = calculateCrossings(xyz1[:crit_num], get_inds=True)
            xyz0_new = sort_to_start(xyz0, inds)
            xyz1_new = sort_to_start(xyz1, inds)
            crit_num = len(inds)
            block_lengths = [(crit_num, crit_num)] + block_lengths
        cost1, f1, g1, theta1 = registration_cost_with_rotation(xyz0_new, xyz1_new, block_lengths=block_lengths, known_theta=known_theta)
        return (cost1, f1, g1, theta1)
    else:
        cost1, f1, g1, theta1, = registration_cost_with_rotation(xyz0, xyz1, block_lengths=block_lengths, known_theta=known_theta)
        return cost1, f1, g1, theta1


def sort_to_start(array, inds):
    non_crit_pts = []
    crit_pts = []
    for i in range(len(array)):
        if i in inds:
            crit_pts.append(array[i])
        else:
            non_crit_pts.append(array[i])
    return np.vstack([np.array(crit_pts), np.array(non_crit_pts)])


def find_closest_auto_with_crossings(demofiles, new_xyz, init_tfm=None, n_jobs=3, seg_proximity=2, DS_LEAF_SIZE=0.02):
    """
    sim_seg_num   : is the index of the segment being executed in the simulation: used to find
    seg_proximity : only segments with numbers in +- seg_proximity of sim_seg_num are selected.
    """
    if args.parallel:
        from joblib import Parallel, delayed

    if not isinstance(demofiles, list):
        demofiles = [demofiles]

    new_xyz = Globals.sim.rope.GetControlPoints() # not sure why this is an issue
    #new_xyz = clouds.downsample(new_xyz,DS_LEAF_SIZE)

    demo_clouds = []
    crit_points = []
    keys = {}
    is_finalsegs = {}
    demotype_num = 0
    seg_num = 0      ## this seg num is the index of all the segments in all the demos.

    for demofile in demofiles:
        equiv = calculateMdp(demofile)
        for demo_name in demofile:
            if demo_name == "demo00041":
                break
            if demo_name != "ar_demo":
                if 'done' in demofile[demo_name].keys():
                    final_seg_id = len(demofile[demo_name].keys()) - 2
                else:
                    final_seg_id = len(demofile[demo_name].keys()) - 1

                for seg_name in demofile[demo_name]:
                    # if demo_name == "demo00010" and seg_name == "seg02": continue
                    if demo_name == "demo00026" and seg_name == "seg02": continue #tip brushes too close above rope
                    if demo_name == "demo00028" and seg_name == "seg02": continue #grabs outside/past the end of the rope
                    if demo_name == "demo00030" and seg_name == "seg02": continue #completely off, grabs way above rope (recording error?)
                    # if demo_name == "demo00039" and seg_name == "seg02": continue
                    if demo_name == "demo00017" and seg_name == "seg00": continue #extra grab
                    if demo_name == "demo00025" and seg_name == "seg01": continue #non-grasping hand is on table, bumps rope
                    if demo_name == "demo00039" and seg_name == "seg02": continue #extra grab
                    #if demo_name != "demo00027" and demo_name != "demo00003": continue

                    if args.choose_demo and demo_name != args.choose_demo: continue
                    if args.choose_seg and seg_name != args.choose_seg: continue #tkl

                    seg_group = demofile[demo_name][seg_name]
                    if 'labeled_points' in seg_group.keys():
                        sim_xyzc, sim_pattern = get_labeled_rope_sim(new_xyz, get_pattern=True)
                        demo_xyzc, demo_pattern = get_labeled_rope_demo(seg_group, get_pattern=True)
                        sim_crossings_inds = [ind for ind in range(len(sim_xyzc)) if sim_xyzc[ind][-1]!=0]
                        if sim_pattern == demo_pattern and sim_pattern == demo_pattern[::-1] and sim_pattern != []:
                            print "symmetric crossings pattern,", demo_pattern

                            keys[seg_num] = (demotype_num, demo_name, seg_name)
                            if seg_name == "seg%02d"%(final_seg_id):
                                is_finalsegs[seg_num] = True
                            else:
                                is_finalsegs[seg_num] = False
                            demo_xyz1 = segment_demo(new_xyz, seg_group, demofile)
                            demo_xyz1 = demo_xyz1.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
                            crit_pts_demo1 = demo_xyz1[sim_crossings_inds] 
                            demo_clouds.append(demo_xyz1)
                            crit_points.append(crit_pts_demo1)
                            seg_num += 1

                            keys[seg_num] = (demotype_num, demo_name, seg_name)
                            if seg_name == "seg%02d"%(final_seg_id):
                                is_finalsegs[seg_num] = True
                            else:
                                is_finalsegs[seg_num] = False
                            demo_xyz2 = segment_demo(new_xyz, seg_group, demofile, reverse=True)
                            demo_xyz2 = demo_xyz2.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
                            crit_pts_demo2 = demo_xyz2[sim_crossings_inds] #hack
                            demo_clouds.append(demo_xyz2) #tkl
                            crit_points.append(crit_pts_demo2)
                            seg_num += 1

                        else:
                            keys[seg_num] = (demotype_num, demo_name, seg_name)
                            if seg_name == "seg%02d"%(final_seg_id):
                                is_finalsegs[seg_num] = True
                            else:
                                is_finalsegs[seg_num] = False                           
                            demo_xyz = segment_demo(new_xyz, seg_group, demofile)
                            if init_tfm is not None:
                                demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
                            demo_clouds.append(demo_xyz)
                            crit_pts_demo = demo_xyz[sim_crossings_inds]
                            crit_points.append(crit_pts_demo)
                            seg_num += 1

                    else:
                        demo_xyz = clouds.downsample(np.asarray(seg_group["cloud_xyz"]),DS_LEAF_SIZE)
                        if init_tfm is not None:
                            try:
                                demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
                            except Exception as exc:
                                print exc
                                import IPython; IPython.embed()

        demotype_num +=1
    if args.parallel:
        print "measuring cost with registration_cost_crit"
        crit_pts_sim = get_critical_points_sim(new_xyz)
        results = Parallel(n_jobs=n_jobs,verbose=51)(delayed(registration_cost_crit)(demo_clouds[i], new_xyz, crit_points[i], crit_pts_sim) for i in keys.keys())
        costs, tfms, tfm_invs, thetas = zip(*results)
    else:
        costs = []
        tfms = []
        tfm_invs = []
        thetas = []

        for (i,ds_cloud) in enumerate(demo_clouds):
            crit_pts_sim = get_critical_points_sim(new_xyz)
            (demotype_num, demo_name, seg_name) = keys[i]
            cost_i, tfm_i, tfm_inverse_i, theta_i = registration_cost_crit(demo_clouds[i], new_xyz, crit_points[i], crit_pts_sim)
            costs.append(cost_i)
            tfms.append(tfm_i)
            tfm_invs.append(tfm_inverse_i)
            thetas.append(theta_i)
            
            print "completed %i/%i"%(i+1, len(demo_clouds))

    print "costs\n", costs

    choice_ind = np.argmin(costs)

    crit_pts_sim = get_critical_points_sim(new_xyz)
    (demotype_num, demo_name, seg_name) = keys[choice_ind]
    seg_group = demofiles[0][demo_name][seg_name]
    
    demo_xyz, demo_lens = fuzz_cloud(demo_clouds[choice_ind], seg_group, init_tfm)
    sim_xyz, sim_lens = fuzz_cloud(new_xyz)
    # import IPython; IPython.embed()
    cost, tfm,_,_ = registration_cost_crit(demo_xyz, sim_xyz, crit_points[choice_ind], crit_pts_sim, zip(demo_lens, sim_lens), thetas[choice_ind])
    print cost
    return (keys[choice_ind][1],keys[choice_ind][2]) , is_finalsegs[choice_ind], tfm, (new_xyz, demo_clouds[choice_ind])


def segment_demo(sim_xyz, seg_group, demofile, reverse=False):
    sim_xyzc, sim_pattern = get_labeled_rope_sim(sim_xyz, get_pattern=True)
    demo_xyzc, demo_pattern = get_labeled_rope_demo(seg_group, get_pattern=True)
    #print seg_group

    #if str(seg_group) == "<HDF5 group \"/demo00015/seg00\" (12 members)>":
        #import IPython; IPython.embed()

    if 1 not in sim_xyzc[:,-1] or 1 not in demo_xyzc[:,-1]: #no crossings in simulation
        #print "no crossings, just unif resample"
        demo_xyz = rope_initialization.unif_resample(demo_xyzc[:,:-1],len(sim_xyz)) #resample to smooth it out
        return demo_xyz

    if sim_pattern != demo_pattern and sim_pattern != demo_pattern[::-1]:
        if len(sim_pattern) == len(demo_pattern):
            print "somethins up"
        #print "could not make patterns match, just unif resample"
        demo_xyz = rope_initialization.unif_resample(demo_xyzc[:,:-1],len(sim_xyz)) #resample to smooth it out
        return demo_xyz

    if sim_pattern != demo_pattern:
        assert sim_pattern == demo_pattern[::-1]
        #print "reversed demo rope"
        demo_xyzc = demo_xyzc[::-1]
        demo_pattern = demo_pattern[::-1]

    if sim_pattern == demo_pattern and sim_pattern == demo_pattern[::-1] and reverse:
        demo_xyzc = demo_xyzc[::-1]

    #print "found pattern match"

    segs = get_rope_segments(demo_xyzc)

    if len(segs) != len(sim_pattern)+1:
        print "\n\nwrong number of segments\n\n"
        import IPython; IPython.embed()

    new_segs = []
    sim_crossings_inds = [ind for ind in range(len(sim_xyzc)) if sim_xyzc[ind][-1]!=0]
    try:
        seg_xyz = rope_initialization.unif_resample(segs[0], sim_crossings_inds[0]+1)[:-1]
        new_segs.append(seg_xyz)
        demo_xyz = seg_xyz
        for i in range(1,len(sim_crossings_inds)):
            seg_xyz = rope_initialization.unif_resample(segs[i], sim_crossings_inds[i]-sim_crossings_inds[i-1]+1)[:-1]
            #seg_xyz[-1] = 0.9*seg_xyz[-1]+0.1*seg_xyz[-2]
            #cur_seg = [0.9*demo_xyz[i]+0.1*demo_xyz[i+1]]
            new_segs.append(seg_xyz)
            demo_xyz = np.vstack([demo_xyz, seg_xyz])
        seg_xyz = rope_initialization.unif_resample(segs[-1], len(sim_xyz)-sim_crossings_inds[-1])#+1)[1:]
        new_segs.append(seg_xyz)
        demo_xyz = np.vstack([demo_xyz, seg_xyz])
    except Exception as exc:
        import IPython; IPython.embed()

    if len(demo_xyz) != len(sim_xyz):
        print "\n\narray lengths do not match\n\n"
        import IPython; IPython.embed()
    #if not args.no_display:
    #    plot_transform_mlab(demo_xyz, sim_xyz, demo_xyz, [np.array(seg) for seg in new_segs])
    if sim_pattern != demo_pattern:
        import IPython; IPython.embed()

    return np.array(demo_xyz)


def get_rope_segments(demo_xyzc):
    segs = []; cur_seg = []
    demo_xyz = demo_xyzc[:,:-1]
    for i in range(len(demo_xyz)):
        cur_seg.append(demo_xyz[i])
        if demo_xyzc[i][-1] != 0: #point is a crossing
            segs.append(cur_seg)
            cur_seg = [demo_xyz[i]+[0, 0, -0.01*demo_xyzc[i][-1]]]
        if i == len(demo_xyz)-1: #last point
            segs.append(cur_seg)
            if len(cur_seg) == 1:
                import IPython; IPython.embed()
    return segs


def plot_transform_mlab(demo_pointcloud, sim_xyz, orig_demo, critical_points_arrays=[], colors=[]):
    from mayavi import mlab; mlab.figure(0, size=(700,650)); mlab.clf()
    plt = mlab.points3d(demo_pointcloud[:,0], demo_pointcloud[:,1], demo_pointcloud[:,2], color=(0,1,0), scale_factor=0.005) #with true/good init tfm
    plt = mlab.points3d(sim_xyz[:,0], sim_xyz[:,1], sim_xyz[:,2], np.linspace(100,110,len(sim_xyz)), scale_factor=0.0001)
    plt = mlab.points3d(orig_demo[:,0], orig_demo[:,1], orig_demo[:,2], np.linspace(100,110,len(orig_demo)), scale_factor=0.00005)
    plt = mlab.points3d(sim_xyz[0,0], sim_xyz[0,1], sim_xyz[0,2], color=(0,1,1), scale_factor=0.02)   #start point of the simulated rope
    color_ind = 0
    for array in critical_points_arrays:
        if colors == []: colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1), (1,1,1), (0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0)]
        plt = mlab.points3d(array[:,0], array[:,1], array[:,2], color=colors[color_ind], scale_factor=0.015)
        color_ind += 1

def plot_interactive(sim_xyz, arrays, colors=[]):
    from hd_visualization import mayavi_plotter as myp; from mayavi import mlab
    p = myp.PlotterInit()
    req = myp.gen_mlab_request(mlab.points3d, sim_xyz[:,0], sim_xyz[:,1], sim_xyz[:,2], np.linspace(100,110,len(sim_xyz)), scale_factor=0.0001)
    p.request(req)
    color_ind = 0
    for array in arrays:
        if colors == []: colors = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (0,1,1), (1,0,1), (1,1,1), (0,0,0)]
        req = myp.gen_mlab_request(mlab.points3d, array[:,0], array[:,1], array[:,2], color=colors[color_ind], scale_factor=0.015)
        p.request(req)
        color_ind += 1

def plot_crossings_2d(critical_points, seg_group):
    from matplotlib import pyplot as plt
    critical_points = np.array(critical_points)
    demo_points = seg_group['crossings'][:,:2]
    labeled_points = seg_group['labeled_points'][:,:2]
    implot = plt.imshow(seg_group['rgb'])
    plt.plot(demo_points[:,0], demo_points[:,1], 'go')
    plt.plot(labeled_points[:,0], labeled_points[:,1], 'wo')
    plt.plot(critical_points[:,0], critical_points[:,1], 'ro')
    plt.show()


def append_to_dict_list(dic, key, item):
    if key in dic.keys():
        dic[key].append(item)
    else:
        dic[key] = [item]

def tpsrpm_plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f):
    ypred_nd = f.transform_points(x_nd)
    handles = []
    if not args.no_display:
        handles.append(Globals.env.plot3(ypred_nd, 3, (0,1,0)))
    #handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, x_nd.min(axis=0), x_nd.max(axis=0), xres = .1, yres = .1, zres = .04))
    if Globals.viewer:
        Globals.viewer.Step()

def arm_moved(hmat_traj):
    return True
    if len(hmat_traj) < 2:
        return False
    tts = hmat_traj[:,:3,3]
    return ((tts[1:] - tts[:-1]).ptp(axis=0) > .01).any()



def slerp(p0, p1, t):
        omega = np.arccos(np.dot(p0/nlg.norm(p0), p1/nlg.norm(p1)))
        so = np.sin(omega)
        return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1


def lerp (x, xp, fp, first=None):
    """
    Returns linearly interpolated n-d vector at specified times.
    """

    fp = np.asarray(fp)

    fp_interp = np.empty((len(x),0))
    for idx in range(fp.shape[1]):
        if first is None:
            interp_vals = np.atleast_2d(np.interp(x,xp,fp[:,idx])).T
        else:
            interp_vals = np.atleast_2d(np.interp(x,xp,fp[:,idx],left=first[idx])).T
        fp_interp = np.c_[fp_interp, interp_vals]

    return fp_interp



def unif_resample(traj, max_diff, wt = None):
    """
    Resample a trajectory so steps have same length in joint space
    """
    import scipy.interpolate as si
    tol = .005
    if wt is not None:
        wt = np.atleast_2d(wt)
        traj = traj*wt

    dl = mu.norms(traj[1:] - traj[:-1],1)
    l = np.cumsum(np.r_[0,dl])
    goodinds = np.r_[True, dl > 1e-8]

    deg = min(3, sum(goodinds) - 1)
    if deg < 1: return traj, np.arange(len(traj))

    nsteps = max(int(np.ceil(float(l[-1])/max_diff)),2)
    newl = np.linspace(0,l[-1],nsteps)

    ncols = traj.shape[1]
    colstep = 10
    traj_rs = np.empty((nsteps,ncols))
    for istart in xrange(0, traj.shape[1], colstep):
        (tck,_) = si.splprep(traj[goodinds, istart:istart+colstep].T,k=deg,s = tol**2*len(traj),u=l[goodinds])
        traj_rs[:,istart:istart+colstep] = np.array(si.splev(newl,tck)).T
    if wt is not None: traj_rs = traj_rs/wt

    newt = np.interp(newl, l, np.arange(len(traj)))

    return traj_rs, newt

def close_traj(traj):

    assert len(traj) > 0

    curr_angs = traj[0]
    new_traj = []

    for i in xrange(len(traj)):
        new_angs = traj[i]
        for j in range(len(new_angs)):
            new_angs[j] = closer_ang(new_angs[j], curr_angs[j])
        new_traj.append(new_angs)
        curr_angs = new_angs

    return new_traj


def downsample_objects(objs, factor):
    """
    Downsample a list of objects based on factor.
    Could streamize this.
    factor needs to be int.
    """
    factor = int(round(factor))
    l = len(objs)
    return objs[0:l:factor]

use_diff_length = args.use_diff_length

def main():
    np.set_printoptions(precision=5, suppress=True)

    global use_diff_length

    if use_diff_length:
        from glob import glob
        demotype_dirs = glob(osp.join(demo_files_dir, args.demo_type+'[0-9]*'))
        demo_types    = [osp.basename(demotype_dir) for demotype_dir in demotype_dirs]
        demo_h5files  = [osp.join(demotype_dir, demo_type+".h5") for demo_type in demo_types]
        print demo_h5files
        demofiles = [h5py.File(demofile, 'r') for demofile in demo_h5files]
        demofile = demofiles[0]
        if len(demofiles) == 1:
            use_diff_length = False
    else:
        demotype_dir = osp.join(demo_files_dir, args.demo_type)
        demo_h5file = osp.join(demotype_dir, args.demo_type+".h5")
        print demo_h5file
        demofile = h5py.File(demo_h5file, 'r')

    if args.select == "clusters":
        if use_diff_length:
            c_h5files = [osp.join(demotype_dir, demo_type+"_clusters.h5") for demo_type in demo_types]
            clusterfiles = [h5py.File(c_h5file, 'r') for c_h5file in c_h5files]
        else:
            c_h5file = osp.join(demotype_dir, args.demo_type+"_clusters.h5")
            clusterfile = h5py.File(c_h5file, 'r')

    Globals.env = openravepy.Environment()
    Globals.env.StopSimulation()
    Globals.sim = ropesim_floating.FloatingGripperSimulation(Globals.env)
    trajoptpy.SetInteractive(args.interactive)
    if not args.no_display:
        Globals.viewer = trajoptpy.GetViewer(Globals.env)

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

    '''
    Add table
    '''
    # As found from measuring
    if not args.remove_table:
        Globals.env.Load(osp.join(cad_files_dir, 'table_sim.xml'))
        body = Globals.env.GetKinBody('table')
        if Globals.viewer:
            Globals.viewer.SetTransparency(body,0.4)
        Globals.table_height = Globals.env.GetKinBody('table').GetLinks()[0].GetGeometries()[0].GetTransform()[2, 3]


    curr_step = 0

    while True:

        if args.max_steps_before_failure != -1 and curr_step > args.max_steps_before_failure:
            redprint("Number of steps %d exceeded maximum %d" % (curr_step, args.max_steps_before_failure))
            break


        curr_step += 1
        '''
        Acquire point cloud
        '''
        redprint("Acquire point cloud")


        rope_cloud = None

        #Set home position in sim
        move_sim_arms_to_side()

        if curr_step > 1:
            # for following steps in rope simulation, using simulation result
            new_xyz = Globals.sim.observe_cloud(upsample=3)
            new_xyz = clouds.downsample(new_xyz, args.cloud_downsample)

            hitch = Globals.env.GetKinBody('hitch')

            if hitch != None:
                pos = hitch.GetTransform()[:3,3]
                hitch_height = hitch_body.GetLinks()[0].GetGeometries()[0].GetCylinderHeight()
                pos[2] = pos[2] - hitch_height/2
                hitch_cloud = cloud_proc_funcs.generate_hitch_points(pos)
                hitch_cloud = clouds.downsample(hitch_cloud, args.cloud_downsample*2)
                new_xyz = np.r_[new_xyz, hitch_cloud]

        else:

            if args.fake_rope:
                rope_nodes = pickle_load(args.fake_rope)
            elif args.init_perturb:
                init_h5file = osp.join(init_state_perturbs_dir, args.demo_type+"_perturb.h5")
                print init_h5file
                init_demofile = h5py.File(init_h5file, 'r')
                #import IPython; IPython.embed()
                fake_xyz = np.squeeze(init_demofile[args.fake_data_demo][args.init_perturb]["cloud_xyz"])
                hmat = openravepy.matrixFromAxisAngle(args.fake_data_transform[3:6])
                hmat[:3,3] = args.fake_data_transform[0:3]
                if args.use_ar_init: hmat = init_tfm.dot(hmat)
                fake_xyz = fake_xyz.dot(hmat[:3,:3].T) + hmat[:3,3][None,:]
                rope_nodes = rope_initialization.find_path_through_point_cloud(fake_xyz)
            else:
                fake_seg = demofile[args.fake_data_demo][args.fake_data_segment]
                fake_xyz = np.squeeze(fake_seg["cloud_xyz"])
                hmat = openravepy.matrixFromAxisAngle(args.fake_data_transform[3:6])
                hmat[:3,3] = args.fake_data_transform[0:3]
                if args.use_ar_init: hmat = init_tfm.dot(hmat)
                # if not rope simulation
                fake_xyz = fake_xyz.dot(hmat[:3,:3].T) + hmat[:3,3][None,:]

                # if the first step in rope simulation
                # rope_nodes = rope_initialization.find_path_through_point_cloud(fake_xyz)
                rope_nodes = get_labeled_rope_demo(fake_seg)
                rope_nodes = rope_initialization.unif_resample(rope_nodes[:,:-1],67)
                rope_nodes = rope_nodes.dot(init_tfm[:3,:3])+init_tfm[:3,3][None,:]

            Globals.sim.create(rope_nodes)
            fake_xyz = Globals.sim.observe_cloud(upsample=3)
            fake_xyz = clouds.downsample(fake_xyz, args.cloud_downsample)
            
            if Globals.viewer:
                rope_body = Globals.env.GetKinBody('rope')
                Globals.viewer.SetTransparency(rope_body,0.6)
#                     print new_xyz.shape
#                     raw_input()


            hitch = Globals.env.GetKinBody('hitch')
            if hitch != None:
                pos = hitch.GetTransform()[:3,3]
                hitch_height = hitch_body.GetLinks()[0].GetGeometries()[0].GetCylinderHeight()
                pos[2] = pos[2] - hitch_height/2
                hitch_cloud = cloud_proc_funcs.generate_hitch_points(pos)
                hitch_cloud = clouds.downsample(hitch_cloud, args.cloud_downsample*2)
                fake_xyz = np.r_[fake_xyz, hitch_cloud]

            new_xyz = fake_xyz

        if args.closest_rope_hack:
            rope_cloud = Globals.sim.observe_cloud(upsample=3)
            rope_cloud = clouds.downsample(rope_cloud, args.cloud_downsample)
        
        '''
        Finding closest demonstration
        '''
        redprint("Finding closest demonstration")
        if args.select=="manual":
            (demo_name, seg_name), is_final_seg = find_closest_manual(demofile)
        elif args.select=="auto":
            (demo_name, seg_name), is_final_seg, f, (simxyz, demoxyz) = find_closest_auto_with_crossings(demofile, new_xyz, init_tfm=init_tfm)

        seg_info = demofile[demo_name][seg_name]
        redprint("closest demo: %s, %s"%(demo_name, seg_name))

        if "done" == seg_name:
            redprint("DONE!")
            break

        '''
        Generating end-effector trajectory
        '''
        redprint("Generating end-effector trajectory")

        handles = []
        old_xyz = np.squeeze(seg_info["cloud_xyz"])
        if args.use_ar_init:
            # Transform the old clouds approximately into PR2's frame
            old_xyz = old_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]

        new_xyz = Globals.sim.rope.GetControlPoints()# #tkl
        old_xyz = demoxyz

        color_old = [(1,0,0,1) for _ in range(len(old_xyz))]
        color_new = [(0,0,1,1) for _ in range(len(new_xyz))]
        color_old_transformed = [(0,1,0,1) for _ in range(len(old_xyz))]
        if not args.no_display:
            handles.append(Globals.env.plot3(old_xyz,10,np.array(color_old)))
            handles.append(Globals.env.drawlinestrip(old_xyz,5, (1,1,0,1)))
            handles.append(Globals.env.plot3(new_xyz,10,np.array(color_new)))
            handles.append(Globals.env.drawlinestrip(new_xyz,5, (0,1,1,1)))

        if not args.no_display:
            handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, old_xyz.min(axis=0)-np.r_[0,0,.1], old_xyz.max(axis=0)+np.r_[0,0,.1], xres = .1, yres = .1, zres = .04))
            handles.append(Globals.env.plot3(f.transform_points(old_xyz),5,np.array(color_old_transformed)))

        orig_eetraj = {}
        eetraj = {}
        for lr in 'lr':
            old_ee_traj = np.asarray(seg_info[lr]["tfms_s"])

            if args.use_ar_init:
                for i in xrange(len(old_ee_traj)):
                    old_ee_traj[i] = init_tfm.dot(old_ee_traj[i])

            new_ee_traj = f.transform_hmats(np.asarray(old_ee_traj))

            eetraj[lr] = new_ee_traj
            orig_eetraj[lr] = old_ee_traj#

            ltraj, rtraj = get_finger_trajs(lr, orig_eetraj[lr])#

            new_ltraj = f.transform_hmats(np.asarray(ltraj))#
            new_rtraj = f.transform_hmats(np.asarray(rtraj))#

        '''
        Generating mini-trajectory
        '''
        miniseg_starts, miniseg_ends, lr_open = split_trajectory_by_gripper(seg_info, args.pot_threshold)
        success = True
        redprint("mini segments: %s %s"%(miniseg_starts, miniseg_ends))

        segment_len = miniseg_ends[-1] - miniseg_starts[0] + 1
        portion = max(args.early_stop_portion, miniseg_ends[0] / float(segment_len))

        #Globals.viewer.Idle() #for recording
        
        for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):

            redprint("Generating joint trajectory for demo %s segment %s, part %i"%(demo_name, seg_name, i_miniseg))
            ### adaptive resampling based on xyz in end_effector
            end_trans_trajs = np.zeros([i_end+1-i_start, 6])
            joint_angles = {}

            print "about to grab"

            old_sim_xyz = Globals.sim.rope.GetControlPoints()#

            for lr in 'lr':
                gripper_open = lr_open[lr][i_miniseg]
                prev_gripper_open = lr_open[lr][i_miniseg-1] if i_miniseg != 0 else False
                if not set_gripper_sim(lr, gripper_open, prev_gripper_open):
                    redprint("Grab %s failed"%lr)
                    success = False
                    file_inc = 0
                    while os.path.isfile("test_results/grab_failed_"+str(file_inc)):
                        file_inc += 1
                    pickle_dump(old_sim_xyz, "test_results/grab_failed_"+str(file_inc))
                    print "Grab failed. Saved preceding state as test_results/grab_failed_"+str(file_inc)
                    import IPython; IPython.embed()
                    #raise Exception("grab failed")

            print "about to calculate"

            for lr in 'lr':
                is_open_miniseg = lr_open[lr][i_miniseg]#
                ltraj, rtraj = get_finger_trajs(lr, orig_eetraj[lr][i_start:i_end+1], is_open_miniseg)#
                new_ltraj = f.transform_hmats(np.asarray(ltraj))#
                new_rtraj = f.transform_hmats(np.asarray(rtraj))#

                if not args.no_display:
                    handles.append(Globals.env.drawlinestrip(old_ee_traj[:,:3,3], 2, (1,0,0,1)))
                    handles.append(Globals.env.drawlinestrip(new_ee_traj[:,:3,3], 2, (0,1,0,1)))
                    handles.append(Globals.env.drawlinestrip(rtraj[:,:3,3], 2, (0,0,1,1)))#
                    handles.append(Globals.env.drawlinestrip(ltraj[:,:3,3], 2, (0,0,1,1)))#
                    handles.append(Globals.env.drawlinestrip(new_rtraj[:,:3,3], 2, (1,0,0,1)))#
                    handles.append(Globals.env.drawlinestrip(new_ltraj[:,:3,3], 2, (1,0,0,1)))#

                joint_angles[lr] = joint_angles_from_fingers(new_ltraj, new_rtraj, is_open_miniseg)#

                for i in xrange(i_start,i_end+1):
                    try:
                        if i==0:
                            prev = resampling.interp_hmats([1],[0,2], np.vstack([new_ltraj[i-i_start:i-i_start+1], new_rtraj[i-i_start:i-i_start+1]]))[0]
                        else:
                            prev = eetraj[lr][i-1]
                        avg = resampling.interp_hmats([1],[0,2], np.vstack([new_ltraj[i-i_start:i-i_start+1], new_rtraj[i-i_start:i-i_start+1]]))[0]
                        eetraj[lr][i] = tooltip_from_fingers(lr, prev, new_ltraj[i-i_start], new_rtraj[i-i_start], joint_angles[lr][i-i_start]) #rework to find tt_tfm that minimizes distance from tips to tfmd pts
                    except Exception as exc:
                        print exc
                        import IPython; IPython.embed()
                    if lr == 'l':
                        end_trans_trajs[i-i_start, :3] = eetraj[lr][i][:3,3]
                    else:
                        end_trans_trajs[i-i_start, 3:] = eetraj[lr][i][:3,3]

                handles.append(Globals.env.drawlinestrip(eetraj[lr][:,:3,3], 2, (1,0,1,1)))

            if not args.no_traj_resample:
                adaptive_times, end_trans_trajs = resampling.adaptive_resample2(end_trans_trajs, 0.001)
            else:
                adaptive_times = range(len(end_trans_trajs))

            miniseg_traj = {}
            for lr in 'lr':
                ee_hmats = resampling.interp_hmats(adaptive_times, range(i_end+1-i_start), eetraj[lr][i_start:i_end+1])
                if arm_moved(ee_hmats):
                    miniseg_traj[lr] = ee_hmats


            safe_drop = {'l': True, 'r': True}
            for lr in 'lr':
                next_gripper_open = lr_open[lr][i_miniseg+1] if i_miniseg < len(miniseg_starts) - 1 else False
                gripper_open = lr_open[lr][i_miniseg] 
                
                if next_gripper_open and not gripper_open:
                    tfm = miniseg_traj[lr][-1]
                    if tfm[2,3] > Globals.table_height + 0.15:
                        safe_drop[lr] = False
                           
            if not (safe_drop['l'] and safe_drop['r']):
                for lr in 'lr':
                    if not safe_drop[lr]:
                        tfm = miniseg_traj[lr][-1]
                        for i in range(1, 8):
                            safe_drop_tfm = tfm
                            safe_drop_tfm[2,3] = tfm[2,3] - i / 10. * (tfm[2,3] - Globals.table_height - 0.15)
                            miniseg_traj[lr].append(safe_drop_tfm)
                    else:
                        for i in range(1, 8):
                            miniseg_traj[lr].append(miniseg_traj[lr][-1])

            #len_miniseg = len(adaptive_times)
            #import IPython; IPython.embed()
            
            redprint("Executing joint trajectory for demo %s segment %s, part %i using arms '%s'"%(demo_name, seg_name, i_miniseg, miniseg_traj.keys()))

            if len(miniseg_traj) > 0:
                """HACK
                """
                is_final_seg = False
                if is_final_seg and miniseg_ends[i_miniseg] < portion * segment_len:
                    success &= exec_traj_sim(miniseg_traj)
                elif is_final_seg:
                    if miniseg_starts[i_miniseg] > portion * segment_len:
                        pass
                    else:
                        sub_traj = {}
                        for lr in miniseg_traj:
                            sub_traj[lr] = miniseg_traj[lr][: int(portion * len(miniseg_traj[lr]))]
                        success &= exec_traj_sim(sub_traj)
                else:
                    print "about to execute"
                    success &= exec_traj_sim(miniseg_traj, ljoints=joint_angles['l'], rjoints=joint_angles['r'])


            Globals.sim.settle(tol=0.0001, animate=True)
            if Globals.viewer and args.interactive:
                Globals.viewer.Idle()
            Globals.sim.settle(tol=0.0001, animate=True)
            Globals.sim.settle(tol=0.0001, animate=True)

        redprint("Demo %s Segment %s result: %s"%(demo_name, seg_name, success))
        
        if args.test_success:
            if isKnot(Globals.sim.rope.GetControlPoints(), True):
                greenprint("Demo %s Segment %s success: isKnot returns true"%(args.fake_data_demo, args.fake_data_segment))
                return
            elif curr_step > 5:
                redprint("Demo %s Segment %s failed: took more than 4 segments"%(args.fake_data_demo, args.fake_data_segment))
                print "too many segments"
                raise Exception("too many segments")
            else:
                rope = Globals.sim.observe_cloud(upsample=5)
                if rope[0][2] < 0:
                    redprint("Demo %s Segment %s failed: rope fell off table"%(args.fake_data_demo, args.fake_data_segment))
                    print "Rope fell off table"
                    raise Exception("Rope fell off table")

    if use_diff_length:
        for demofile in demofiles: demofile.close()
        if args.select == "clusters":
            for clusterfile in clusterfiles: clusterfile.close()
    else:
        demofile.close()
        if args.select == "clusters":
            clusterfile.close()

if __name__ == "__main__":
    main()