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


parser.add_argument("--pot_threshold",type=float, default=15)

parser.add_argument("--use_ar_init", action="store_true", default=False)
parser.add_argument("--use_base", action="store_true", default=False)
parser.add_argument("--not_allow_base", help="dont allow base movement when use_base", action="store_true", default=False)
parser.add_argument("--early_stop_portion", help="stop early in the final segment to avoid bullet simulation problem", type=float, default=0.5)
parser.add_argument("--no_traj_resample", action="store_true", default=False)

parser.add_argument("--interactive",action="store_true", default=False)
parser.add_argument("--remove_table", action="store_true")
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--no_display",action="store_true", default=False)


parser.add_argument("--friction", help="friction value in bullet", type=float, default=1.0)

parser.add_argument("--max_steps_before_failure", type=int, default=-1)
parser.add_argument("--tps_bend_cost_init", type=float, default=1)
parser.add_argument("--tps_bend_cost_final", type=float, default=.001)
parser.add_argument("--tps_bend_cost_final_search", type=float, default=.00001)
parser.add_argument("--tps_n_iter", type=int, default=50)

parser.add_argument("--closest_rope_hack", action="store_true", default=False)
parser.add_argument("--closest_rope_hack_thresh", type=float, default=0.01)
parser.add_argument("--cloud_downsample", type=float, default=.01)

parser.add_argument("--use_crossings", action="store_true", default=False)
parser.add_argument("--use_rotation", action="store_true", default=False)
parser.add_argument("--test_success", action="store_true", default=False)

parser.add_argument("--choose_demo", nargs='+', type=str)
parser.add_argument("--choose_seg", nargs='+', type=str)


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
import cPickle
import numpy as np, numpy.linalg as nlg
import math

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
        tfm_head_dof, tfm_bf_head, cad_files_dir
from knot_classifier import calculateCrossings, calculateMdp, isKnot



L_POSTURES = {'side': np.array([[-0.98108876, -0.1846131 ,  0.0581623 ,  0.10118172],
                                [-0.19076337,  0.97311662, -0.12904799,  0.68224057],
                                [-0.03277475, -0.13770277, -0.98993119,  0.91652485],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]]) }

R_POSTURES = {'side' : np.array([[-0.98108876,  0.1846131 ,  0.0581623 ,  0.10118172],
                                 [ 0.19076337,  0.97311662,  0.12904799, -0.68224057],
                                 [-0.03277475,  0.13770277, -0.98993119,  0.91652485],
                                 [ 0.        ,  0.        ,  0.        ,  1.        ]]) }

crossings_to_demos = {} #dictionary matching crossings patterns to demo ids
global_demo_clouds = []
global_keys = {}
global_is_finalsegs = {}

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

    print rgrip
    print lgrip

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

"""
Not sure if these are required.
"""
def rotate_about_median(xyz, theta):
    """
    rotates xyz by theta around the median along the x, y dimensions
    """
    median = np.median(xyz, axis=0)
    centered_xyz = xyz - median
    if np.shape(theta):
        r_mat = theta
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
    aligned_xyz = aligned_xyz.dot(np.linalg.pinv(pca1)) # aligned to PC axes of xyz1   
    new_xyz = aligned_xyz + median
    return new_xyz

def get_critical_points(rope_points, seg_group):
    _, crossings_locations = calculateCrossings(sim_xyz)
    crit_pts1 = np.vstack(rope_points[0,:], crossings_locations, rope_points[-1,:])
    demo_pts = np.empty((len(seg_group["crossings"])+2,3))
    for i in range(seg_group["crossings"]):
        demo_pts[i+1,:2] = seg_group["crossings"][:2]
        demo_pts[i+1,2] = crit_pts1[i+1,2] #hacky guess at z -- or should use avg from seg ptcloud? probably yes
        clouds.xyZ_to_XY() #get closest point in demo_cloud


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
        Globals.sim.step()
        if Globals.viewer:
            Globals.viewer.Step()
            if args.interactive: Globals.viewer.Idle()
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

def exec_traj_sim(lr_traj, animate=True):
    def sim_callback(i):
        Globals.sim.step()
    if args.no_display:
        animate = False

    lhmats_up, rhmats_up = ropesim_floating.retime_hmats(lr_traj['l'], lr_traj['r'])

    # in simulation mode, we must make sure to gradually move to the new starting position
    curr_rtf  = Globals.sim.grippers['r'].get_toolframe_transform()
    curr_ltf  = Globals.sim.grippers['l'].get_toolframe_transform()

    l_transition_hmats, r_transition_hmats = ropesim_floating.retime_hmats([curr_ltf, lhmats_up[0]], [curr_rtf, rhmats_up[0]])

    animate_traj.animate_floating_traj(l_transition_hmats, r_transition_hmats,
                                       Globals.sim, pause=False,
                                       callback=sim_callback, step_viewer=animate, step=args.step)
    animate_traj.animate_floating_traj(lhmats_up, rhmats_up, Globals.sim, pause=False,
                                       callback=sim_callback, step_viewer=animate, step=args.step)
    return True

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


def registration_cost(xyz0, xyz1):
    scaled_xyz0, _ = registration.unit_boxify(xyz0)
    scaled_xyz1, _ = registration.unit_boxify(xyz1)
    f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, reg_init=args.tps_bend_cost_init, reg_final = args.tps_bend_cost_final_search, rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=30)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    return cost

def registration_cost_and_tfm(xyz0, xyz1):
    scaled_xyz0, src_params = registration.unit_boxify(xyz0)
    scaled_xyz1, targ_params = registration.unit_boxify(xyz1)
    f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, reg_init=args.tps_bend_cost_init, reg_final = args.tps_bend_cost_final_search, rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=30)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    g = registration.unscale_tps_3d(g, targ_params, src_params)
    f = registration.unscale_tps_3d(f, src_params, targ_params)
    return (cost, f, g)

def registration_cost_with_rotation(xyz0, xyz1):
    rad_angs = np.linspace(-np.pi+np.pi*(float(10)/180), np.pi,36)
    #rad_angs = np.linspace(-np.pi+np.pi*(float(30)/180), np.pi,12)
    #rad_angs = np.linspace(-np.pi+np.pi*(float(45)/180), np.pi,8)
    costs = np.zeros(len(rad_angs))
    for i in range(len(rad_angs)):
        try:
            rotated_demo = rotate_about_median(xyz0, rad_angs[i])
        except ValueError as e:
            import IPython
            IPython.embed()
        scaled_rotated_demo, src_params = registration.unit_boxify(rotated_demo)
        scaled_xyz1, targ_params = registration.unit_boxify(xyz1)
        f,g = registration.tps_rpm_bij(scaled_rotated_demo, scaled_xyz1, reg_init=args.tps_bend_cost_init, reg_final = args.tps_bend_cost_final_search, rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=3)
        costs[i] = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    theta = rad_angs[np.argmin(costs)]
    rotated_demo = rotate_about_median(xyz0, theta)
    scaled_rotated_demo, src_params = registration.unit_boxify(rotated_demo)
    scaled_xyz1, targ_params = registration.unit_boxify(xyz1)
    f,g = registration.tps_rpm_bij(scaled_rotated_demo, scaled_xyz1, reg_init=args.tps_bend_cost_init, reg_final = args.tps_bend_cost_final_search, rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=30)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g) #plus some rotation penalty?
    g = registration.unscale_tps_3d(g, targ_params, src_params)
    f = registration.unscale_tps_3d(f, src_params, targ_params)

    rotation_matrix = np.eye(3)
    rotation_matrix_inv = np.eye(3)
    rotation_matrix[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    rotation_matrix_inv[0:2, 0:2] = np.array([[np.cos(-theta), -np.sin(-theta)],[np.sin(-theta), np.cos(-theta)]])
    shift_from_origin = Affine(np.eye(3), np.median(xyz0, axis=0))
    rotate = Affine(rotation_matrix, np.zeros((3,)))
    unrotate = Affine(rotation_matrix_inv, np.zeros((3,)))
    shift_to_origin = Affine(np.eye(3), -1*np.median(xyz0, axis=0))
    f = Composition([shift_to_origin, rotate, shift_from_origin, f])
    g = Composition([g, shift_to_origin, unrotate, shift_from_origin])

    return (cost, f, g, theta)

def registration_cost_with_pca_rotation(xyz0, xyz1):
    costs = []; tfms = []
    pca_demo = rotate_by_pca(xyz0, xyz1)
    ptclouds = [pca_demo, reflect_across_median(pca_demo, 1), rotate_about_median(pca_demo, np.pi)]
    for ptcloud in ptclouds:
        scaled_xyz1, targ_params = registration.unit_boxify(xyz1)
        scaled_xyz0, src_params = registration.unit_boxify(ptcloud)
        f1,g1 = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, reg_init=args.tps_bend_cost_init, reg_final = args.tps_bend_cost_final_search, rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=3)
        costs.append(registration.tps_reg_cost(f1) + registration.tps_reg_cost(g1))
        f = registration.unscale_tps_3d(f1, src_params, targ_params)
        g = registration.unscale_tps_3d(g1, targ_params, src_params)
        tfms.append((f,g))
    f,g = tfms[np.argmin(costs)]
    pca0,_,_ = np.linalg.svd(np.cov(xyz0.T)); pca1,_,_ = np.linalg.svd(np.cov(xyz1.T))
    rmat = pca0.dot(np.linalg.pinv(pca1))
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


def find_closest_auto(demofiles, new_xyz, init_tfm=None, n_jobs=3, seg_proximity=2, DS_LEAF_SIZE=0.02):
    """
    sim_seg_num   : is the index of the segment being executed in the simulation: used to find
    seg_proximity : only segments with numbers in +- seg_proximity of sim_seg_num are selected.
    """
    if args.parallel:
        from joblib import Parallel, delayed

    if not isinstance(demofiles, list):
        demofiles = [demofiles]

    demo_clouds = []


    new_xyz = clouds.downsample(new_xyz,DS_LEAF_SIZE)

    avg = 0.0

    keys = {}
    is_finalsegs = {}
    demotype_num = 0
    seg_num = 0      ## this seg num is the index of all the segments in all the demos.
    for demofile in demofiles:
        for demo_name in demofile:
            if demo_name != "ar_demo":
                if 'done' in demofile[demo_name].keys():
                    final_seg_id = len(demofile[demo_name].keys()) - 2
                else:
                    final_seg_id = len(demofile[demo_name].keys()) - 1

                for seg_name in demofile[demo_name]:
                    if args.choose_demo and demo_name not in args.choose_demo:
                        continue
                    if args.choose_seg and seg_name not in args.choose_seg:
                        continue

                    keys[seg_num] = (demotype_num, demo_name, seg_name)

                    if seg_name == "seg%02d"%(final_seg_id):
                        is_finalsegs[seg_num] = True
                    else:
                        is_finalsegs[seg_num] = False

                    seg_num += 1
                    demo_xyz = clouds.downsample(np.asarray(demofile[demo_name][seg_name]["cloud_xyz"]),DS_LEAF_SIZE)
                    if init_tfm is not None:
                        demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
                    print demo_xyz.shape
                    avg += demo_xyz.shape[0]
    
                    demo_clouds.append(demo_xyz)

        demotype_num +=1

    # raw_input(avg/len(demo_clouds))
    if args.parallel:
        costs = Parallel(n_jobs=n_jobs,verbose=51)(delayed(registration_cost)(demo_cloud, new_xyz) for demo_cloud in demo_clouds)
    else:
        costs = []
        for (i,ds_cloud) in enumerate(demo_clouds):
            cost_i = registration_cost(ds_cloud, new_xyz)
            costs.append(cost_i)
            print "completed %i/%i"%(i+1, len(demo_clouds))

    print "costs\n", costs

    if args.show_neighbors:
        nshow = min(5, len(demo_clouds.keys()))
        import cv2, hd_rapprentice.cv_plot_utils as cpu
        sortinds = np.argsort(costs)[:nshow]

        near_rgbs = []
        for i in sortinds:
            (demo_name, seg_name) = keys[i]
            near_rgbs.append(np.asarray(demofile[demo_name][seg_name]["rgb"]))

        bigimg = cpu.tile_images(near_rgbs, 1, nshow)
        cv2.imshow("neighbors", bigimg)
        print "press any key to continue"
        cv2.waitKey()

    choice_ind = np.argmin(costs)

    if demotype_num == 1:
        return (keys[choice_ind][1],keys[choice_ind][2]) , is_finalsegs[choice_ind]
    else:
        return keys[choice_ind], is_finalsegs[choice_ind]



def find_closest_auto_with_crossings(demofiles, new_xyz, init_tfm=None, n_jobs=3, seg_proximity=2, DS_LEAF_SIZE=0.02):
    """
    sim_seg_num   : is the index of the segment being executed in the simulation: used to find
    seg_proximity : only segments with numbers in +- seg_proximity of sim_seg_num are selected.
    """
    if args.parallel:
        from joblib import Parallel, delayed

    if not isinstance(demofiles, list):
        demofiles = [demofiles]

    new_xyz = clouds.downsample(new_xyz,DS_LEAF_SIZE)

    global global_keys
    global global_demo_clouds
    global global_is_finalsegs    

    if global_keys == {}: #need to initialize keys, demo_clouds, is_finalsegs
        demo_clouds = []
        keys = {}
        is_finalsegs = {}
        demotype_num = 0
        seg_num = 0      ## this seg num is the index of all the segments in all the demos.

        for demofile in demofiles:
            equiv = calculateMdp(demofile)
            if crossings_to_demos == {}:
                fill_crossings = True
            else:
                fill_crossings = False
            for demo_name in demofile:
                if demo_name != "ar_demo":
                    if 'done' in demofile[demo_name].keys():
                        final_seg_id = len(demofile[demo_name].keys()) - 2
                    else:
                        final_seg_id = len(demofile[demo_name].keys()) - 1

                    for seg_name in demofile[demo_name]:

                        keys[seg_num] = (demotype_num, demo_name, seg_name)

                        points = []
                        for crossing in demofile[demo_name][seg_name]["crossings"]:
                            points.append(crossing[2])
                        equivalent_states = []
                        if tuple(points) in equiv:
                            equivalent_states = equiv[tuple(points)]
                        elif tuple(points[::-1]) in equiv:
                            equivalent_states = equiv[tuple(points)]
                        for state in equivalent_states:
                            if state in crossings_to_demos:
                                if seg_num not in crossings_to_demos[state]:
                                    crossings_to_demos[state].append(seg_num)
                            else:
                                crossings_to_demos[state] = [seg_num]
                            if state[::-1] in crossings_to_demos:
                                if seg_num not in crossings_to_demos[state[::-1]]:
                                    crossings_to_demos[state[::-1]].append(seg_num)
                            else:
                                crossings_to_demos[state[::-1]] = [seg_num]

                        if seg_name == "seg%02d"%(final_seg_id):
                            is_finalsegs[seg_num] = True
                        else:
                            is_finalsegs[seg_num] = False

                        seg_num += 1
                        demo_xyz = clouds.downsample(np.asarray(demofile[demo_name][seg_name]["cloud_xyz"]),DS_LEAF_SIZE)
                        if init_tfm is not None:
                            demo_xyz = demo_xyz.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
                        print demo_xyz.shape
                        
                        demo_clouds.append(demo_xyz)

            demotype_num +=1
        #save keys, demo_clouds, is_finalsegs globally so they can be used again on future segments
        global_keys = keys
        global_demo_clouds = demo_clouds
        global_is_finalsegs = is_finalsegs
    else:
        sim_xyz = Globals.sim.rope.GetControlPoints()
        sim_crossings, crossings_locations = calculateCrossings(sim_xyz)
        sim_crossings = tuple(sim_crossings)
        i = 0
        keys = {}
        demo_clouds = []
        is_finalsegs = {}
        if sim_crossings in crossings_to_demos:
            for seg_num in crossings_to_demos[sim_crossings]:
                keys[i] = global_keys[seg_num]
                demo_clouds.append(global_demo_clouds[seg_num])
                is_finalsegs[i] = global_is_finalsegs[seg_num]
                i += 1
        else:
            keys = global_keys
            demo_clouds = global_demo_clouds
            is_finalsegs = global_is_finalsegs
    if args.parallel:
        if args.use_rotation:
            results = Parallel(n_jobs=n_jobs,verbose=51)(delayed(registration_cost_with_pca_rotation)(demo_cloud, new_xyz) for demo_cloud in demo_clouds)
            costs, tfms, tfm_invs, thetas = zip(*results)
        else:
            results = Parallel(n_jobs=n_jobs,verbose=51)(delayed(registration_cost_and_tfm)(demo_cloud, new_xyz) for demo_cloud in demo_clouds)
            costs, tfms, tfm_invs = zip(*results)
            thetas = []
    else:
        costs = []
        tfm_invs = []
        thetas = []
        
        for (i,ds_cloud) in enumerate(demo_clouds):
            if args.use_rotation:
                cost_i, tfm_inverse_i , theta_i = registration_cost_with_pca_rotation(ds_cloud, new_xyz)
                thetas.append(theta_i)
            else:
                cost_i, tfm_inverse_i = registration_cost_and_tfm(ds_cloud, new_xyz)                    
            costs.append(cost_i)
            tfm_invs.append(tfm_inverse_i)
            
            print "completed %i/%i"%(i+1, len(demo_clouds))

    print "costs\n", costs

    choice_ind = match_crossings(demofiles, keys, costs, tfms, tfm_invs, init_tfm, demo_clouds, thetas)

    #for now, assume only one demotype at a time
    return (keys[choice_ind][1],keys[choice_ind][2]) , is_finalsegs[choice_ind], tfms[choice_ind]

def match_crossings(demofiles, keys, costs, tfms, tfm_invs, init_tfm, dclouds, thetas):
    import scipy
    from scipy.spatial import *
    
    print "matching crossings"
    sim_xyz = Globals.sim.rope.GetControlPoints()
    sim_crossings, crossings_locations = calculateCrossings(sim_xyz)
    #^ avoid using downsampled cloud as points are "out of order".
    cost_inds = np.argsort(costs)
    for demofile in demofiles:
        equiv = calculateMdp(demofile)
        if tuple(sim_crossings) not in equiv and tuple(reversed(sim_crossings)) not in equiv:
            print sim_crossings, "not in equiv"
            return np.argmin(costs)
    for choice_ind in cost_inds: #check best TPS fit against crossings match
        demotype_num, demo, seg = keys[choice_ind]
        hdf = demofiles[demotype_num]
        demo_pattern = []
        demo_points = []
        for crossing in hdf[demo][seg]["crossings"]:
            demo_pattern.append(crossing[2])
            demo_points.append(crossing[:2])

        if args.use_rotation:
            demo_pointcloud = rotate_about_median(dclouds[choice_ind], thetas[choice_ind])
        else:
            demo_pointcloud = dclouds[choice_ind]
        orig_demo = (dclouds[choice_ind] - init_tfm[:3,3][None,:]).dot(np.linalg.pinv(init_tfm[:3,:3]).T)

        critical_points = np.vstack([sim_xyz[0], crossings_locations, sim_xyz[-1]])
#        critical_points = np.vstack([sim_xyz[0], sim_xyz[-1]])
        critical_points = tfm_invs[choice_ind].transform_points(critical_points)
        simrope_start = critical_points[0]
        simrope_finish = critical_points[-1]
        critical_points = (critical_points - init_tfm[:3,3][None,:]).dot(np.linalg.pinv(init_tfm[:3,:3]).T)
        init_inverse_start = critical_points[0]
        init_inverse_finish = critical_points[-1]
        xystart = clouds.XYZ_to_xy(*critical_points[0])
        xyfinish = clouds.XYZ_to_xy(*critical_points[-1])

        flat_points = []
        for point in critical_points:
            flat_points.append(clouds.XYZ_to_xy(*point))

        plot_crossings_2d(flat_points, hdf[demo][seg])

        if hdf[demo][seg]["ends"] and hdf[demo][seg]["ends"].shape[0] > 0:   #ends info exists and is not empty
            ends_cost = np.hypot(*(hdf[demo][seg]["ends"][:,:2] - xystart).T) + np.hypot(*((hdf[demo][seg]["ends"][:,:2] - xyfinish)[::-1]).T)
            
            points_cost = [0,0]
            b = hdf[demo][seg]["ends"][:,:2]
            m = np.array([demo_points, demo_points[::-1]])
            e = b[::-1]
            demo_points2 = np.array([np.vstack([b[0], m[0,:], e[0]]), np.vstack([b[1], m[1,:], e[1]])])
            if len(demo_points2[0]) != len(flat_points):
                print "array of critical points in demonstration is different length from simulation"
                # import IPython
                # IPython.embed()
            else:
                for i in range(len(demo_points2[0])):
                    points_cost[0] += np.hypot(*(demo_points2[0,i] - flat_points[i]))
                    points_cost[1] += np.hypot(*(demo_points2[1,i] - flat_points[i]))

            if np.ma.min(ends_cost) > 40:
                continue
            if np.ma.min(points_cost) > (len(demo_points)*20):
                print "points cost too high"
                # import IPython
                # IPython.embed()
                # continue
            if np.argmin(ends_cost) != 0:
                print "swapped ends"
                #continue //just give up on this demo?
                demo_pattern.reverse()

        if not args.no_display:
            plot_transform_mlab(demo_pointcloud, sim_xyz, orig_demo, init_inverse_start, init_inverse_finish, critical_points)

        if demo_pattern == sim_crossings:
            print "match found"
            return choice_ind
        
        equiv = calculateMdp(hdf)
        if tuple(demo_pattern) in equiv:
            equivalent_states = equiv[tuple(demo_pattern)]
            if tuple(sim_crossings) in equivalent_states:
                return choice_ind
            print "sim_crossings not in equiv[demo_pattern]"
        else:
            print "pattern not found in equiv"

        print "discarding initial choice - did not match crossings pattern"
        print "demonstration pattern:", tuple(demo_pattern)
        print "sim_crossings pattern:", tuple(sim_crossings)
        print "choice ind:", str(choice_ind)+"/"+str(len(cost_inds))

    return np.argmin(costs)


def plot_transform_mlab(demo_pointcloud, sim_xyz, orig_demo, simrope_start, simrope_finish, critical_points):
    from mayavi import mlab; mlab.figure(0, size=(700,650)); mlab.clf()
    plt = mlab.points3d(demo_pointcloud[:,0], demo_pointcloud[:,1], demo_pointcloud[:,2], color=(0,1,0), scale_factor=0.01) #with true/good init tfm
    plt = mlab.points3d(sim_xyz[:,0], sim_xyz[:,1], sim_xyz[:,2], np.linspace(100,110,len(sim_xyz)), scale_factor=0.00005)
    plt = mlab.points3d(orig_demo[:,0], orig_demo[:,1], orig_demo[:,2], np.linspace(100,110,len(orig_demo)), scale_factor=0.00005)
    plt = mlab.points3d(sim_xyz[0,0], sim_xyz[0,1], sim_xyz[0,2], color=(0,1,0), scale_factor=0.03)   #endpoint of the simulated rope
    plt = mlab.points3d(simrope_start[0],simrope_start[1],simrope_start[2], color=(0,0,1), scale_factor=0.03)  #points given by tps
    plt = mlab.points3d(simrope_finish[0], simrope_finish[1], simrope_finish[2], color=(1,1,0), scale_factor=0.03)
    plt = mlab.points3d(critical_points[:,0], critical_points[:,1], critical_points[:,2], color=(1,0,0), scale_factor=0.04)

def plot_crossings_2d(critical_points, seg_group):
    from matplotlib import pyplot as plt
    critical_points = np.array(critical_points)
    demo_points = seg_group['crossings'][:,:2]
    implot = plt.imshow(seg_group['rgb'])
    plt.plot(critical_points[:,0], critical_points[:,1], 'ro')
    plt.plot(demo_points[:,0], demo_points[:,1], 'go')
    plt.show()

def append_to_dict_list(dic, key, item):
    if key in dic.keys():
        dic[key].append(item)
    else:
        dic[key] = [item]

def find_closest_clusters(demofiles, clusterfiles, new_xyz, sim_seg_num, seg_proximity=2, init_tfm=None, check_n=3, n_jobs=3, DS_LEAF_SIZE=0.02):
    if args.parallel:
        from joblib import Parallel, delayed

    if not isinstance(demofiles, list): demofiles = [demofiles]
    if not isinstance(clusterfiles, list): clusterfiles = [clusterfiles]

    new_xyz = clouds.downsample(new_xyz,DS_LEAF_SIZE)

    # Store all the best cluster clouds
    clusters = {}
    keys = {}
    cluster_clouds = []
    all_keys = {}
    idx = 0
    dnum = 0
    for demofile,clusterfile in zip(demofiles, clusterfiles):
        keys[dnum] = clusterfile['keys']
        keys[dnum] = {int(key):keys[dnum][key] for key in keys[dnum]}
        clusters[dnum] = clusterfile['clusters']
        clusters[dnum] = {int(c):clusters[dnum][c] for c in clusters[dnum]}

        for cluster in clusters[dnum]:
            best_seg = clusters[dnum][cluster][0]
            dname, sname = keys[dnum][best_seg]
            cloud = clouds.downsample(np.asarray(demofile[dname][sname]["cloud_xyz"]),DS_LEAF_SIZE)
            if init_tfm is not None:
                cloud = cloud.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]

            all_keys[idx] = (dnum, cluster)
            cluster_clouds.append(cloud)
            idx += 1
        dnum += 1

    # Check the clusters with min costs
    if args.parallel:
        ccosts = Parallel(n_jobs=n_jobs,verbose=51)(delayed(registration_cost)(cloud, new_xyz) for cloud in cluster_clouds)
    else:
        ccosts = []
        for (i,ds_cloud) in enumerate(cluster_clouds):
            ccosts.append(registration_cost(ds_cloud, new_xyz))
            print "completed %i/%i"%(i+1, len(cluster_clouds))

    print "Cluster costs: \n", ccosts


    best_clusters = np.argsort(ccosts)
    check_n = min(check_n, len(best_clusters))

    if args.show_neighbors:
        nshow = min(check_n*3, len(cluster_clouds))
        import cv2, hd_rapprentice.cv_plot_utils as cpu
        closeinds = best_clusters[:nshow]

        near_rgbs = []
        for i in closeinds:
            dn, cluster = all_keys[i]
            (demo_name, seg_name) = keys[dn][clusters[dn][cluster][0]]
            near_rgbs.append(np.asarray(demofiles[dn][demo_name][seg_name]["rgb"]))

        rows = 6
        cols = int(math.ceil(nshow*1.0/rows))
        bigimg = cpu.tile_images(near_rgbs, rows, cols, max_width=300)
        cv2.imshow("neighbors", bigimg)
        print "press any key to continue"
        cv2.waitKey()

    #############################################################################
    demo_seg_clouds = {}
    demo_seg_info   = {}

    def is_final_seg(seg_info):
        dn, dname, sname = seg_info
        if 'done' in demofiles[dn][dname].keys():
            final_seg_id = len(demofiles[dn][dname].keys()) - 2
        else:
            final_seg_id = len(demofiles[dn][dname].keys()) - 1
        return sname == "seg%02d"%(final_seg_id)

    for c in best_clusters[:check_n]:
        dn, cluster = all_keys[c]
        cluster_segs = clusters[dn][cluster]

        for seg in cluster_segs:
            dname,sname = keys[dn][seg]

            snum  = int(sname.split('seg')[1])
            sdist = int(np.abs(sim_seg_num - snum)//seg_proximity)

            cloud = clouds.downsample(np.asarray(demofiles[dn][dname][sname]["cloud_xyz"]),DS_LEAF_SIZE)
            if init_tfm is not None:
                cloud = cloud.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]

            append_to_dict_list(demo_seg_clouds, sdist, cloud)
            append_to_dict_list(demo_seg_info, sdist, (dn,dname,sname))


    smallest_seg_dist = np.sort(demo_seg_clouds.keys())[0]
    check_clouds = demo_seg_clouds[smallest_seg_dist]
    cluster_keys = demo_seg_info[smallest_seg_dist]
    # Check the clusters with min costs
    if args.parallel:
        costs = Parallel(n_jobs=n_jobs,verbose=51)(delayed(registration_cost)(cloud, new_xyz) for cloud in check_clouds)
    else:
        costs = []
        for (i,ds_cloud) in enumerate(check_clouds):
            costs.append(registration_cost(ds_cloud, new_xyz))
            print "completed %i/%i"%(i+1, len(check_clouds))

    print "Costs: \n", costs

    if args.show_neighbors:
        nshow = min(30, len(check_clouds))
        sortinds = np.argsort(costs)[:nshow]

        near_rgbs = []
        for i in sortinds:
            (dn, demo_name, seg_name) = cluster_keys[i]
            near_rgbs.append(np.asarray(demofiles[dn][demo_name][seg_name]["rgb"]))

        rows = 6
        cols = int(math.ceil(nshow*1.0/rows))
        bigimg = cpu.tile_images(near_rgbs, rows, cols, max_width=1000)
        cv2.imshow("neighbors2", bigimg)
        print "press any key to continue"
        cv2.waitKey()

    choice_ind = np.argmin(costs)

    if dnum == 1:
        return (cluster_keys[choice_ind][1],cluster_keys[choice_ind][2]) , is_final_seg(cluster_keys[choice_ind])
    else:
        return cluster_keys[choice_ind], is_final_seg(cluster_keys[choice_ind])



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
    import pdb
    pdb.set_trace()
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

def has_hitch(h5data, demo_name=None, seg_name=None):
    if demo_name != None and seg_name != None:
        return "hitch_pos" in h5data[demo_name][seg_name].keys()
    else:
        first_demo = h5data[h5data.keys()[0]]
        first_seg = first_demo[first_demo.keys()[0]]
        return "hitch_pos" in first_seg


use_diff_length = args.use_diff_length


def main():
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


    # if has_hitch(demofile, args.fake_data_demo, args.fake_data_segment):
    #     Globals.env.Load(osp.join(cad_files_dir, 'hitch.xml'))
    #     hitch_pos = demofile[args.fake_data_demo][args.fake_data_segment]['hitch_pos']
    #     hitch_body = Globals.env.GetKinBody('hitch')
    #     table_body = Globals.env.GetKinBody('table')
    #     if init_tfm != None:
    #         hitch_pos = init_tfm[:3,:3].dot(hitch_pos) + init_tfm[:3,3]
    #     hitch_tfm = hitch_body.GetTransform()
    #     hitch_tfm[:3, 3] = hitch_pos
    #     hitch_height = hitch_body.GetLinks()[0].GetGeometries()[0].GetCylinderHeight()
    #     table_z_extent = table_body.GetLinks()[0].GetGeometries()[0].GetBoxExtents()[2]
    #     table_height = table_body.GetLinks()[0].GetGeometries()[0].GetTransform()[2, 3]
    #     hitch_tfm[2, 3] = table_height + table_z_extent + hitch_height/2.0
    #     hitch_body.SetTransform(hitch_tfm)

    '''
    Add table
    '''
    # As found from measuring
    if not args.remove_table:
        Globals.env.Load(osp.join(cad_files_dir, 'table_sim.xml'))
        body = Globals.env.GetKinBody('table')
        if Globals.viewer:
            Globals.viewer.SetTransparency(body,0.4)


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
            new_xyz = Globals.sim.observe_cloud(3)
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

            fake_seg = demofile[args.fake_data_demo][args.fake_data_segment]
            fake_xyz = np.squeeze(fake_seg["cloud_xyz"])
            hmat = openravepy.matrixFromAxisAngle(args.fake_data_transform[3:6])
            hmat[:3,3] = args.fake_data_transform[0:3]
            if args.use_ar_init: hmat = init_tfm.dot(hmat)
            # if not rope simulation
            fake_xyz = fake_xyz.dot(hmat[:3,:3].T) + hmat[:3,3][None,:]


            # if the first step in rope simulation
            rope_nodes = rope_initialization.find_path_through_point_cloud(fake_xyz)
            Globals.sim.create(rope_nodes)
            fake_xyz = Globals.sim.observe_cloud(3)
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
            rope_cloud = Globals.sim.observe_cloud(3)
            rope_cloud = clouds.downsample(rope_cloud, args.cloud_downsample)
        
        '''
        Finding closest demonstration
        '''
        redprint("Finding closest demonstration")
        if use_diff_length:
            if args.select=="manual":
                dnum, (demo_name, seg_name), is_final_seg = find_closest_manual(demofiles)
            elif args.select=="auto":
#                 dnum, (demo_name, seg_name), is_final_seg = find_closest_auto(demofiles, new_xyz, init_tfm=init_tfm, DS_LEAF_SIZE = args.cloud_downsample)
                if args.use_crossings:
                    dnum, (demo_name, seg_name), is_final_seg, f = find_closest_auto_with_crossings(demofiles, new_xyz, init_tfm=init_tfm, DS_LEAF_SIZE = args.cloud_downsample)
                else:
                    dnum, (demo_name, seg_name), is_final_seg = find_closest_auto(demofiles, new_xyz, init_tfm=init_tfm, DS_LEAF_SIZE = args.cloud_downsample)
            else:
                dnum, (demo_name, seg_name), is_final_seg = find_closest_clusters(demofiles, clusterfiles, new_xyz, curr_step-1, init_tfm=init_tfm, DS_LEAF_SIZE = args.cloud_downsample)

            seg_info = demofiles[dnum][demo_name][seg_name]
            redprint("closest demo: %i, %s, %s"%(dnum, demo_name, seg_name))
        else:
            if args.select=="manual":
                (demo_name, seg_name), is_final_seg = find_closest_manual(demofile)
            elif args.select=="auto":
#                 (demo_name, seg_name), is_final_seg = find_closest_auto(demofile, new_xyz, init_tfm=init_tfm)
                if args.use_crossings:
                    (demo_name, seg_name), is_final_seg, f = find_closest_auto_with_crossings(demofile, new_xyz, init_tfm=init_tfm)
                else:
                    (demo_name, seg_name), is_final_seg = find_closest_auto(demofile, new_xyz, init_tfm=init_tfm)
            else:
                (demo_name, seg_name), is_final_seg = find_closest_clusters(demofile, clusterfile, new_xyz, curr_step-1, init_tfm=init_tfm)
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

        #print old_xyz.shape
        #print new_xyz.shape

        color_old = [(1,0,0,1) for _ in range(len(old_xyz))]
        color_new = [(0,0,1,1) for _ in range(len(new_xyz))]
        color_old_transformed = [(0,1,0,1) for _ in range(len(old_xyz))]
        if not args.no_display:
            handles.append(Globals.env.plot3(old_xyz,5,np.array(color_old)))
            handles.append(Globals.env.plot3(new_xyz,5,np.array(color_new)))

        t1 = time.time()
        if not args.use_rotation:
            scaled_old_xyz, src_params = registration.unit_boxify(old_xyz)
            scaled_new_xyz, targ_params = registration.unit_boxify(new_xyz)
            f,_ = registration.tps_rpm_bij(scaled_old_xyz, scaled_new_xyz, plot_cb = tpsrpm_plot_cb,
                                           plotting=0,rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=args.tps_n_iter, reg_init=args.tps_bend_cost_init, reg_final=args.tps_bend_cost_final)
            f = registration.unscale_tps(f, src_params, targ_params)
        t2 = time.time()

        if not args.no_display:
            handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, old_xyz.min(axis=0)-np.r_[0,0,.1], old_xyz.max(axis=0)+np.r_[0,0,.1], xres = .1, yres = .1, zres = .04))
            handles.append(Globals.env.plot3(f.transform_points(old_xyz),5,np.array(color_old_transformed)))


        print 'time: %f'%(t2-t1)

        eetraj = {}
        for lr in 'lr':
            old_ee_traj = np.asarray(seg_info[lr]["tfms_s"])

            if args.use_ar_init:
                for i in xrange(len(old_ee_traj)):
                    old_ee_traj[i] = init_tfm.dot(old_ee_traj[i])

            new_ee_traj = f.transform_hmats(np.asarray(old_ee_traj))

            eetraj[lr] = new_ee_traj

            if not args.no_display:
                handles.append(Globals.env.drawlinestrip(old_ee_traj[:,:3,3], 2, (1,0,0,1)))
                handles.append(Globals.env.drawlinestrip(new_ee_traj[:,:3,3], 2, (0,1,0,1)))

        '''
        Generating mini-trajectory
        '''
        miniseg_starts, miniseg_ends, lr_open = split_trajectory_by_gripper(seg_info, args.pot_threshold)
        success = True
        redprint("mini segments: %s %s"%(miniseg_starts, miniseg_ends))

        segment_len = miniseg_ends[-1] - miniseg_starts[0] + 1
        portion = max(args.early_stop_portion, miniseg_ends[0] / float(segment_len))
        
        for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):

            redprint("Generating joint trajectory for demo %s segment %s, part %i"%(demo_name, seg_name, i_miniseg))
            ### adaptive resampling based on xyz in end_effector
            end_trans_trajs = np.zeros([i_end+1-i_start, 6])

            for lr in 'lr':
                for i in xrange(i_start,i_end+1):
                    if lr == 'l':
                        end_trans_trajs[i-i_start, :3] = eetraj[lr][i][:3,3]
                    else:
                        end_trans_trajs[i-i_start, 3:] = eetraj[lr][i][:3,3]


            if not args.no_traj_resample:
                adaptive_times, end_trans_trajs = resampling.adaptive_resample2(end_trans_trajs, 0.001)
            else:
                adaptive_times = range(len(end_trans_trajs))

            miniseg_traj = {}
            for lr in 'lr':
                ee_hmats = resampling.interp_hmats(adaptive_times, range(i_end+1-i_start), eetraj[lr][i_start:i_end+1])
                if arm_moved(ee_hmats):
                    miniseg_traj[lr] = ee_hmats

            #len_miniseg = len(adaptive_times)

            redprint("Executing joint trajectory for demo %s segment %s, part %i using arms '%s'"%(demo_name, seg_name, i_miniseg, miniseg_traj.keys()))
            
            for lr in 'lr':
                gripper_open = lr_open[lr][i_miniseg]
                prev_gripper_open = lr_open[lr][i_miniseg-1] if i_miniseg != 0 else False
                if not set_gripper_sim(lr, gripper_open, prev_gripper_open):
                    redprint("Grab %s failed"%lr)
                    success = False

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
                    success &= exec_traj_sim(miniseg_traj)


        Globals.sim.settle(animate=True)
        if Globals.viewer and args.interactive:
            Globals.viewer.Idle()

        redprint("Demo %s Segment %s result: %s"%(demo_name, seg_name, success))
        
        if args.test_success:
            if isKnot(Globals.sim.rope.GetControlPoints(), True):
                greenprint("Demo %s Segment %s success: isKnot returns true"%(args.fake_data_demo, args.fake_data_segment))
                return
            elif curr_step > 4:
                redprint("Demo %s Segment %s failed: took more than 4 segments"%(args.fake_data_demo, args.fake_data_segment))
                raise Exception("too many segments")
            else:
                rope = Globals.sim.observe_cloud(5)
                if rope[0][2] < 0:
                    redprint("Demo %s Segment %s failed: rope fell off table"%(args.fake_data_demo, args.fake_data_segment))
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

"""
decrease resolution of rotations - maybe 8 is enough. Compare in terms of speed, success
pca for initialization (e.g. align axes)
distance to grasping points
distance to crossings locations
segment labeling? e.g. this segment was in between these two kinds of crossings, now these other two, thus it is this important/unimportant
smooth trajectory in new space -- avoid grasping issues
finer resolution trajectory -- less force?
"""
