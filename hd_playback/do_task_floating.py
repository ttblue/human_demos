#!/usr/bin/env python

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
    
parser.add_argument("--simulation", type=int, default=0)
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
parser.add_argument("--ar_demo_file",type=str, default="")
parser.add_argument("--ar_run_file",type=str, default="")
parser.add_argument("--use_base", action="store_true", default=False)
parser.add_argument("--not_allow_base", help="dont allow base movement when use_base", action="store_true", default=False)
parser.add_argument("--early_stop_portion", help="stop early in the final segment to avoid bullet simulation problem", type=float, default=0.5)
parser.add_argument("--no_traj_resample", action="store_true", default=False)

parser.add_argument("--interactive",action="store_true", default=False)
parser.add_argument("--remove_table", action="store_true")

parser.add_argument("--friction", help="friction value in bullet", type=float, default=1.0)

parser.add_argument("--max_steps_before_failure", type=int, default=-1)
parser.add_argument("--tps_bend_cost_init", type=float, default=1)
parser.add_argument("--tps_bend_cost_final", type=float, default=.00001)
parser.add_argument("--tps_n_iter", type=int, default=50)

parser.add_argument("--closest_rope_hack", action="store_true", default=False)
parser.add_argument("--closest_rope_hack_thresh", type=float, default=0.01)
parser.add_argument("--cloud_downsample", type=float, default=.01)



args = parser.parse_args()

if args.fake_data_segment is None: assert args.execution==1


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
import numpy as np
from numpy.linalg import norm

import openravepy
 
from hd_rapprentice import registration, animate_traj, \
     plotting_openrave, task_execution, \
     planning, resampling, \
     ropesim_floating, rope_initialization
from hd_utils import yes_or_no, clouds, math_utils as mu, cloud_proc_funcs
from hd_utils.pr2_utils import get_kinect_transform
from hd_utils.colorize import *
from hd_utils.utils import avg_transform
from hd_utils.defaults import demo_files_dir, hd_data_dir, asus_xtion_pro_f, \
        ar_init_dir, ar_init_demo_name, ar_init_playback_name, \
        tfm_head_dof, tfm_bf_head, tfm_gtf_ee, cad_files_dir



L_POSTURES = {'side': np.array([[-0.98108876, -0.1846131 ,  0.0581623 ,  0.10118172],
                                [-0.19076337,  0.97311662, -0.12904799,  0.68224057],
                                [-0.03277475, -0.13770277, -0.98993119,  0.91652485],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]]) }

R_POSTURES = {'side' : np.array([[-0.98108876,  0.1846131 ,  0.0581623 ,  0.10118172],
                                 [ 0.19076337,  0.97311662,  0.12904799, -0.68224057],
                                 [-0.03277475,  0.13770277, -0.98993119,  0.91652485],
                                 [ 0.        ,  0.        ,  0.        ,  1.        ]]) }


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
            
     
#     import IPython
#     IPython.embed()
 
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
    r_mat = np.eye(3)
    r_mat[0:2, 0:2] = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    rotated_xyz = centered_xyz.dot(r_mat)
    new_xyz = rotated_xyz + median    
    return new_xyz

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
    closed_angle = (0 if not args.simulation else .02) * mult
    
    target_val = open_angle if is_open else closed_angle

    if is_open and not prev_is_open:
        Globals.sim.release_rope(lr)

    # execute gripper open/close trajectory
    start_val = Globals.sim.grippers[lr].get_gripper_joint_value()
    joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .02))
    for val in joint_traj:
        Globals.sim.grippers[lr].set_gripper_joint_value(val)
        Globals.sim.step()
        if args.animation:
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

def exec_traj_sim(lr_traj, animate):
    def sim_callback(i):
        Globals.sim.step()

    lhmats_up, rhmats_up = ropesim_floating.retime_hmats(lr_traj['l'], lr_traj['r'])

    # in simulation mode, we must make sure to gradually move to the new starting position
    curr_rtf  = Globals.sim.grippers['r'].get_toolframe_transform()
    curr_ltf  = Globals.sim.grippers['l'].get_toolframe_transform()
   
    l_transition_hmats, r_transition_hmats = ropesim_floating.retime_hmats([curr_ltf, lhmats_up[0]], [curr_rtf, rhmats_up[0]])

    animate_traj.animate_floating_traj(l_transition_hmats, r_transition_hmats,
                                       Globals.sim, pause=False,
                                       callback=sim_callback, step_viewer=animate)
    animate_traj.animate_floating_traj(lhmats_up, rhmats_up, Globals.sim, pause=False,
                                       callback=sim_callback, step_viewer=animate)
    return True

def find_closest_manual(demofiles, _new_xyz):
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
    f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, rot_reg=1e-3, n_iter=30)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    return cost


def find_closest_auto(demofiles, new_xyz, sim_seg_num, init_tfm=None, n_jobs=3, seg_proximity=2, DS_LEAF_SIZE=0.02):
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
            costs.append(registration_cost(ds_cloud, new_xyz))
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
        import cv2, hd_rapprentice.cv_plot_utils as cpu, math
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
        cv2.imshow("neighbors", bigimg)
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


#ros stuffs
def mirror_arm_joints(x):
    "mirror image of joints (r->l or l->r)"
    return np.r_[-x[0],x[1],-x[2],x[3],-x[4],x[5],-x[6]]


L_POSTURES = dict(        
    untucked = [0.4,  1.0,   0.0,  -2.05,  0.0,  -0.1,  0.0],
    tucked = [0.06, 1.25, 1.79, -1.68, -1.73, -0.10, -0.09],
    up = [ 0.33, -0.35,  2.59, -0.15,  0.59, -1.41, -0.27],
    side = [  1.832,  -0.332,   1.011,  -1.437,   1.1  ,  -2.106,  3.074]
)   


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
    
    trajoptpy.SetInteractive(args.interactive)
    
    if args.log:
        LOG_DIR = osp.join(demotype_dir, "do_task_logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.mkdir(LOG_DIR)
        LOG_COUNT = 0

    if args.execution:
        rospy.init_node("exec_task",disable_signals=True)
        Globals.pr2 = PR2.PR2()
        Globals.env = Globals.pr2.env
        Globals.robot = Globals.pr2.robot
    else:
        Globals.env = openravepy.Environment()
        Globals.env.StopSimulation()
        Globals.env.Load("robots/pr2-beta-static.zae")
        Globals.robot = Globals.env.GetRobots()[0]
        
        if args.simulation:
            Globals.sim = ropesim.Simulation(Globals.env, Globals.robot, args.friction)
        
    if args.animation:
        Globals.viewer = trajoptpy.GetViewer(Globals.env)
    '''
    Add table
    '''
    # As found from measuring
    if not args.remove_table:
        a= osp.join(cad_files_dir, 'table.xml')
        print a
        if args.execution:
            Globals.env.Load(osp.join(cad_files_dir, 'table.xml'))
        else:
            Globals.env.Load(osp.join(cad_files_dir, 'table_sim.xml'))
        body = Globals.env.GetKinBody('table')
        
        if Globals.viewer:
            Globals.viewer.SetTransparency(body,0.4)


    # get rgbd from pr2?
    if not args.fake_data_segment or not args.fake_data_demo:
        import cloudprocpy
        grabber = cloudprocpy.CloudGrabber()
        grabber.startRGBD()

    init_tfm = None
    if args.use_ar_init:
        # Get ar marker from demo:
        if args.ar_demo_file == "":
            # default demo_file
            ar_demo_file = osp.join(hd_data_dir, ar_init_dir, ar_init_demo_name)
        else:
            ar_demo_file = args.ar_demo_file
        with open(ar_demo_file,'r') as fh: ar_demo_tfms = cPickle.load(fh)
        
        ar_marker = ar_demo_tfms['marker']
        # use camera 1 as default
        ar_marker_cameras = [1]
        ar_demo_tfm = avg_transform([ar_demo_tfms['tfms'][c] for c in ar_demo_tfms['tfms'] if c in ar_marker_cameras])
        
        # Get ar marker for PR2:
        ar_run_tfm = None
        if args.execution:
            try:
                rgb, depth = grabber.getRGBD()
                xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
                pc = xyzrgb2pc(xyz, rgb)
                
                ar_tfms = get_ar_marker_poses(pc, ar_markers=[ar_marker])
                if ar_tfms:
                
                    blueprint("Found ar marker %i for initialization!"%ar_marker)
                    ar_run_tfm = np.asarray(ar_tfms[ar_marker])                
                    # save ar marker found in another file?
                    save_ar = {'marker': ar_marker, 'tfm': ar_run_tfm}
    
                    with open(osp.join(hd_data_dir, ar_init_dir, ar_init_playback_name),'w') as fh: cPickle.dump(save_ar, fh)
                    print "Saved new position." 
                
            except Exception as e:
                yellowprint("Exception: %s"%str(e))
    
        if ar_run_tfm is None:
            if args.ar_run_file == "":
                # default demo_file
                ar_run_file = osp.join(hd_data_dir, ar_init_dir, ar_init_playback_name)
                
            else:
                ar_run_file = args.ar_run_file
            with open(ar_run_file,'r') as fh: ar_run_tfms = cPickle.load(fh)
            
            # use camera 1 as default
            ar_run_tfm = ar_run_tfms['tfm']

        # transform to move the demo points approximately into PR2's frame
        # Basically a rough transform from head kinect to demo_camera, given the tables are the same.
        init_tfm = ar_run_tfm.dot(np.linalg.inv(ar_demo_tfm))
        if args.fake_data_segment and args.fake_data_demo:
            init_tfm = tfm_bf_head.dot(tfm_head_dof).dot(init_tfm)
        else:
            #T_w_k here should be different from rapprentice
            T_w_k = get_kinect_transform(Globals.robot)
            init_tfm = T_w_k.dot(init_tfm)

    if args.fake_data_demo and args.fake_data_segment:

        if has_hitch(demofile, args.fake_data_demo, args.fake_data_segment):
            Globals.env.Load(osp.join(cad_files_dir, 'hitch.xml'))
            hitch_pos = demofile[args.fake_data_demo][args.fake_data_segment]['hitch_pos']
            hitch_body = Globals.env.GetKinBody('hitch')
            table_body = Globals.env.GetKinBody('table')
            if init_tfm != None:
                hitch_pos = init_tfm[:3,:3].dot(hitch_pos) + init_tfm[:3,3]
            hitch_tfm = hitch_body.GetTransform()
            hitch_tfm[:3, 3] = hitch_pos
            hitch_height = hitch_body.GetLinks()[0].GetGeometries()[0].GetCylinderHeight()
            table_z_extent = table_body.GetLinks()[0].GetGeometries()[0].GetBoxExtents()[2] 
            table_height = table_body.GetLinks()[0].GetGeometries()[0].GetTransform()[2, 3]
            hitch_tfm[2, 3] = table_height + table_z_extent + hitch_height/2.0
            hitch_body.SetTransform(hitch_tfm)

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
                   
        if args.simulation:
            #Set home position in sim
            l_vals = L_POSTURES['side']
            Globals.robot.SetDOFValues(l_vals, Globals.robot.GetManipulator('leftarm').GetArmIndices())
            Globals.robot.SetDOFValues(mirror_arm_joints(l_vals), Globals.robot.GetManipulator('rightarm').GetArmIndices())

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
                new_xyz = np.squeeze(fake_seg["cloud_xyz"])
        
                hmat = openravepy.matrixFromAxisAngle(args.fake_data_transform[3:6])
                hmat[:3,3] = args.fake_data_transform[0:3]
                if args.use_ar_init: hmat = init_tfm.dot(hmat)
        
                # if not rope simulation
                new_xyz = new_xyz.dot(hmat[:3,:3].T) + hmat[:3,3][None,:]
                # if the first step in rope simulation
                if args.simulation: # curr_step == 1
                    rope_nodes = rope_initialization.find_path_through_point_cloud(new_xyz)
                    Globals.sim.create(rope_nodes)
                    new_xyz = Globals.sim.observe_cloud(3)
                    new_xyz = clouds.downsample(new_xyz, args.cloud_downsample)
#                     print new_xyz.shape
#                     raw_input()


                    hitch = Globals.env.GetKinBody('hitch')
                    if hitch != None:
                        pos = hitch.GetTransform()[:3,3]
                        hitch_height = hitch_body.GetLinks()[0].GetGeometries()[0].GetCylinderHeight()
                        pos[2] = pos[2] - hitch_height/2
                        hitch_cloud = cloud_proc_funcs.generate_hitch_points(pos)
                        hitch_cloud = clouds.downsample(hitch_cloud, args.cloud_downsample*2)
                        new_xyz = np.r_[new_xyz, hitch_cloud]
                    
            if args.closest_rope_hack:
                if args.simulation:
                    rope_cloud = Globals.sim.observe_cloud(3)
                    rope_cloud = clouds.downsample(rope_cloud, args.cloud_downsample)
                else:
                    if has_hitch(demofile, args.fake_data_demo, args.fake_data_segment):
                        rope_cloud = demofile[args.fake_data_demo][args.fake_data_segment]['object']
                    else:
                        rope_cloud = demofile[args.fake_data_demo][args.fake_data_segment]['cloud_xyz']


        else:
            Globals.pr2.head.set_pan_tilt(0,1.2)
            Globals.pr2.rarm.goto_posture('side')
            Globals.pr2.larm.goto_posture('side')
            Globals.pr2.join_all()
            if args.execution: time.sleep(3.5)
        
            Globals.pr2.update_rave()
            
            rgb, depth = grabber.getRGBD()
            
            new_xyz = cloud_proc_func(rgb, depth, T_w_k)
            new_xyz = clouds.downsample(new_xyz, args.cloud_downsample)
            
            if args.closest_rope_hack:
                rope_cloud = np.array(new_xyz)
            
            if has_hitch(demofile):
                hitch_normal = clouds.clouds_plane(new_xyz)
                
                hitch_cloud, hitch_pos = hitch_proc_func(rgb, depth, T_w_k, hitch_normal)
                hitch_cloud = clouds.downsample(hitch_cloud, args.cloud_downsample*2)
                new_xyz = np.r_[new_xyz, hitch_cloud]
                
            
        
        if args.log:
            LOG_COUNT += 1
            import cv2
            cv2.imwrite(osp.join(LOG_DIR,"rgb%05i.png"%LOG_COUNT), rgb)
            cv2.imwrite(osp.join(LOG_DIR,"depth%05i.png"%LOG_COUNT), depth)
            np.save(osp.join(LOG_DIR,"xyz%i.npy"%LOG_COUNT), new_xyz)
            
            
            
        '''
        Finding closest demonstration
        '''
        redprint("Finding closest demonstration")
        if use_diff_length:
            if args.select=="manual":
                dnum, (demo_name, seg_name), is_final_seg = find_closest_manual(demofiles, new_xyz)
            elif args.select=="auto":
                dnum, (demo_name, seg_name), is_final_seg = find_closest_auto(demofiles, new_xyz, init_tfm, DS_LEAF_SIZE = args.cloud_downsample)
            else:
                dnum, (demo_name, seg_name), is_final_seg = find_closest_clusters(demofiles, clusterfiles, new_xyz, curr_step-1, init_tfm=init_tfm, DS_LEAF_SIZE = args.cloud_downsample)

            seg_info = demofiles[dnum][demo_name][seg_name]
            redprint("closest demo: %i, %s, %s"%(dnum, demo_name, seg_name))
        else:
            if args.select=="manual":
                (demo_name, seg_name), is_final_seg = find_closest_manual(demofile, new_xyz)
            elif args.select=="auto":
                (demo_name, seg_name), is_final_seg = find_closest_auto(demofile, new_xyz, init_tfm)
            else:
                (demo_name, seg_name), is_final_seg = find_closest_clusters(demofile, clusterfile, new_xyz, curr_step-1, init_tfm=init_tfm)
            seg_info = demofile[demo_name][seg_name]
            redprint("closest demo: %s, %s"%(demo_name, seg_name))
        
        if "done" == seg_name:
            redprint("DONE!")
            break

    
        if args.log:
            with open(osp.join(LOG_DIR,"neighbor%i.txt"%LOG_COUNT),"w") as fh: fh.write(seg_name)

        # import matplotlib.pylab as plt
        # plt.plot(np.np.asarray(demofile[demo_name][seg_name]['r']['pot_angles'])[:,0])
        # plt.show()

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
        handles.append(Globals.env.plot3(old_xyz,5,np.array(color_old)))
        handles.append(Globals.env.plot3(new_xyz,5,np.array(color_new)))

        t1 = time.time()
        scaled_old_xyz, src_params = registration.unit_boxify(old_xyz)
        scaled_new_xyz, targ_params = registration.unit_boxify(new_xyz)
        f,_ = registration.tps_rpm_bij(scaled_old_xyz, scaled_new_xyz, plot_cb = tpsrpm_plot_cb,
                                       plotting=5 if args.animation else 0,rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=args.tps_n_iter, reg_init=args.tps_bend_cost_init, reg_final=args.tps_bend_cost_final)
        f = registration.unscale_tps(f, src_params, targ_params)
        t2 = time.time()
        
        handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, old_xyz.min(axis=0)-np.r_[0,0,.1], old_xyz.max(axis=0)+np.r_[0,0,.1], xres = .1, yres = .1, zres = .04))
        
        handles.append(Globals.env.plot3(f.transform_points(old_xyz),5,np.array(color_old_transformed)))
        
        
        print 'time: %f'%(t2-t1)


        eetraj = {}
        for lr in 'lr':
            link_name = "%s_gripper_tool_frame"%lr
            
            old_ee_traj = np.asarray(seg_info[lr]["tfms_s"])
            #old_ee_traj = np.asarray(downsample_objects(seg_info[lr]["tfms_s"], args.downsample))
            
            
            if args.use_ar_init:
                for i in xrange(len(old_ee_traj)):
                    old_ee_traj[i] = init_tfm.dot(old_ee_traj[i])
            new_ee_traj = f.transform_hmats(np.asarray(old_ee_traj))

        
            eetraj[link_name] = new_ee_traj
            
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
        
        prev_vals = {lr:None for lr in 'lr'}
        #l_vals = PR2.Arm.L_POSTURES['side']
        l_vals = L_POSTURES['side']
        for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):
            
            if args.execution =="real": Globals.pr2.update_rave()
            
            redprint("Generating joint trajectory for demo %s segment %s, part %i"%(demo_name, seg_name, i_miniseg))
            
            
            
            ### adaptive resampling based on xyz in end_effector
            end_trans_trajs = np.zeros([i_end+1-i_start, 6])
            
            for lr in 'lr':    
                ee_link_name = "%s_gripper_tool_frame"%lr
                for i in xrange(i_start,i_end+1):
                    if lr == 'l':
                        end_trans_trajs[i-i_start, :3] = eetraj[ee_link_name][i][:3,3]
                    else:
                        end_trans_trajs[i-i_start, 3:] = eetraj[ee_link_name][i][:3,3]
                        
            
            if not args.no_traj_resample:
                adaptive_times, end_trans_trajs = resampling.adaptive_resample2(end_trans_trajs, 0.001)
            else:
                adaptive_times = range(len(end_trans_trajs))
            
            ee_hmats = {}
            for lr in 'lr':
                ee_link_name = "%s_gripper_tool_frame"%lr
                ee_hmats[ee_link_name] = resampling.interp_hmats(adaptive_times, range(i_end+1-i_start), eetraj[ee_link_name][i_start:i_end+1])
                
            len_miniseg = len(adaptive_times)
            
            ### trajopt init traj
            init_joint_trajs = {}
            for lr in 'lr':
                link_name = "%s_gripper_tool_frame"%lr
                if args.trajopt_init == 'all_zero':
                    init_joint_traj = np.zeros((len_miniseg, 7))
                    init_joint_trajs[lr] = init_joint_traj
                    
                elif args.trajopt_init == 'openrave_ik':
                    init_joint_traj = []
                    
                    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
                    manip = Globals.robot.GetManipulator(manip_name)
                    ik_type = openravepy.IkParameterizationType.Transform6D

                    all_x = []
                    x = []

                    for (i, pose_matrix) in enumerate(ee_hmats[link_name]):
                        
                        rot_pose_matrix = pose_matrix.dot(tfm_gtf_ee)
                        sols = manip.FindIKSolutions(openravepy.IkParameterization(rot_pose_matrix, ik_type),
                                                     openravepy.IkFilterOptions.CheckEnvCollisions)
                        
                        
                        all_x.append(i)
                        
                        if sols != []:
                            x.append(i)

                            reference_sol = None
                            for sol in reversed(init_joint_traj):
                                if sol != None:
                                    reference_sol = sol
                                    break
                            if reference_sol is None:
                                if prev_vals[lr] is not None:
                                    reference_sol = prev_vals[lr]
                                else:
                                    reference_sol = l_vals if lr == 'l' else mirror_arm_joints(l_vals)
                        
                            
                            sols = [closer_angs(sol, reference_sol) for sol in sols]
                            norm_differences = [norm(np.asarray(reference_sol) - np.asarray(sol), 2) for sol in sols]
                            min_index = norm_differences.index(min(norm_differences))
                            
                            init_joint_traj.append(sols[min_index])
                            
                            blueprint("Openrave IK succeeds")
                        else:
                            redprint("Openrave IK fails")                        
#                         
#                         if sol != None:
#                             x.append(i)
#                             init_joint_traj.append(sol)
#                             blueprint("Openrave IK succeeds")
#                         else:
#                             redprint("Openrave IK fails")


                    if len(x) == 0:
                        if prev_vals[lr] is not None:
                            vals = prev_vals[lr]
                        else:
                            vals = l_vals if lr == 'l' else mirror_arm_joints(l_vals)
                        
                        init_joint_traj_interp = np.tile(vals,(len_miniseg, 1))
                    else:
                        if prev_vals[lr] is not None:
                            init_joint_traj_interp = lerp(all_x, x, init_joint_traj, first=prev_vals[lr])
                        else:
                            init_joint_traj_interp = lerp(all_x, x, init_joint_traj)
                    
                    yellowprint("Openrave IK found %i solutions out of %i."%(len(x), len(all_x)))
                    
                    init_traj_close = close_traj(init_joint_traj_interp.tolist())
                    init_joint_trajs[lr] = np.asarray(init_traj_close) 

                elif args.trajopt_init == 'trajopt_ik':
                    init_joint_traj = []
                    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
                    manip = Globals.robot.GetManipulator(manip_name)
                    
                    import trajopt_ik

                    all_x = []
                    x = []
                    
                    for (i, pose_matrix) in enumerate(ee_hmats[link_name]):
                        sol = trajopt_ik.inverse_kinematics(Globals.robot, manip_name, link_name, pose_matrix)
                        
                        all_x.append(i)
                        if sol != None:
                            x.append(i)
                            init_joint_traj.append(sol)
                            blueprint("Trajopt IK succeeds")
                        else:
                            redprint("Trajopt IK fails")

                    if prev_vals[lr] is not None:
                        init_joint_traj_interp = lerp(all_x, x, init_joint_traj, first=prev_vals[lr])
                    else:
                        init_joint_traj_interp = lerp(all_x, x, init_joint_traj)

                    yellowprint("Trajopt IK found %i solutions out of %i."%(len(x), len(all_x)))    
                    
                    init_traj_close = close_traj(init_joint_traj_interp.tolist())
                    init_joint_trajs[lr] = np.asarray(init_traj_close)
                    
                else:
                    redprint("trajopt initialization method %s not supported"%(args.trajopt_init))
                    redprint("use default all zero initialization instead")
                    init_joint_traj = np.zeros((len_miniseg, 7))
                    init_joint_trajs[lr] = init_joint_traj
         
         
         
                init_joint_trajs[lr] = unwrap_arm_traj_in_place(init_joint_trajs[lr])
                
            redprint("start generating full body trajectory")

                            
            ### Generate full-body trajectory
            bodypart2traj = {}
            
            if args.use_base:
                
                new_hmats = {}
                init_traj = {}
                
                end_pose_constraints = {}
                
                for lr in 'lr':
                    ee_link_name = "%s_gripper_tool_frame"%lr
                    new_ee_traj = downsample_objects(ee_hmats[ee_link_name], args.downsample)
                
                    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
                    new_hmats[manip_name] = new_ee_traj
                    init_traj[manip_name] = downsample_objects(init_joint_trajs[lr], args.downsample)
 
                    """
                    Dirty hack.
                    """
                    now_val = binarize_gripper(seg_info[lr]["pot_angles"][i_end], args.pot_threshold)
                    next_val = binarize_gripper(seg_info[lr]["pot_angles"][i_end+1], args.pot_threshold)
                    
                    if next_val < now_val:
                        end_pose_constraint = True
                    else:
                        end_pose_constraint = False
                    """
                    End dirty hack.
                    """            
                    
                    end_pose_constraints[manip_name[lr]] = end_pose_constraints
                    
                    
                
                
                active_dofs = np.r_[Globals.robot.GetManipulator("rightarm").GetArmIndices(), Globals.robot.GetManipulator("leftarm").GetArmIndices()]
                
                allow_base = not args.not_allow_base
                if allow_base:
                    Globals.robot.SetActiveDOFs(active_dofs, 
                                                openravepy.DOFAffine.X + openravepy.DOFAffine.Y + openravepy.DOFAffine.RotationAxis,
                                                [0, 0, 1])
                else:
                    Globals.robot.SetActiveDOFs(active_dofs)                
                    
                new_joint_traj, _ = planning.plan_fullbody(Globals.robot, Globals.env, new_hmats, init_traj, allow_base=allow_base)
                
                part_names = {"leftarm":"larm", "rightarm":"rarm", "base":"base"}
                
                for bodypart_name in new_joint_traj:
                    bodypart2traj[part_names[bodypart_name]] = new_joint_traj[bodypart_name]
                    


            else:
                for lr in 'lr':
                    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
    
                    ee_link_name = "%s_gripper_tool_frame"%lr
                    new_ee_traj = ee_hmats[ee_link_name]
                        
                    """
                    Dirty hack.
                    """
                    now_val = binarize_gripper(seg_info[lr]["pot_angles"][i_end], args.pot_threshold)
                    next_val = binarize_gripper(seg_info[lr]["pot_angles"][i_end+1], args.pot_threshold)
                    
                    if next_val < now_val:
                        end_pose_constraint = True
                    else:
                        end_pose_constraint = False
                    """
                    End dirty hack.
                    """
                    
                    new_ee_traj = downsample_objects(new_ee_traj, args.downsample)
                    init_joints = downsample_objects(init_joint_trajs[lr], args.downsample)

                    t1 = time.time()
                    new_joint_traj = planning.plan_follow_traj(Globals.robot, manip_name,
                                                               Globals.robot.GetLink(ee_link_name),
                                                               new_ee_traj, init_joints,
                                                               rope_cloud=rope_cloud,
                                                               rope_constraint_thresh=args.closest_rope_hack_thresh,
                                                               end_pose_constraint=end_pose_constraint)
                    t2 = time.time()
                    print 'time: %f'%(t2-t1)
                    


                    prev_vals[lr] = new_joint_traj[-1]
                    #handles.append(Globals.env.drawlinestrip(new_ee_traj[:,:3,3], 2, (0,1,0,1))
                    
                    
                    part_name = {"l":"larm", "r":"rarm"}[lr]
                    bodypart2traj[part_name] = new_joint_traj
                
                    
 
            
            
                
                       
            if args.execution: Globals.pr2.update_rave()

            redprint("Executing joint trajectory for demo %s segment %s, part %i using arms '%s'"%(demo_name, seg_name, i_miniseg, bodypart2traj.keys()))
            
            for lr in 'lr':
                gripper_open = lr_open[lr][i_miniseg]
                prev_gripper_open = lr_open[lr][i_miniseg-1] if i_miniseg != 0 else False
                if not set_gripper_maybesim(lr, gripper_open, prev_gripper_open):
                    redprint("Grab %s failed"%lr)
                    success = False

                        
            if len(bodypart2traj['larm']) > 0:
                """HACK
                """
                is_final_seg = False
                if is_final_seg and miniseg_ends[i_miniseg] < portion * segment_len:
                    success &= exec_traj_maybesim(bodypart2traj)
                elif is_final_seg:
                    if miniseg_starts[i_miniseg] > portion * segment_len:
                        pass
                    else:
                        sub_bodypart2traj = {}
                        for lr in bodypart2traj:
                            sub_bodypart2traj[lr] = bodypart2traj[lr][: int(portion * len(bodypart2traj[lr]))]
                        success &= exec_traj_maybesim(sub_bodypart2traj)
                else:
                    success &= exec_traj_maybesim(bodypart2traj)
                    
                '''
                Maybe for robot execution
                '''
                if args.execution: time.sleep(5)
                
                if args.execution:
                    time.sleep(5)
 

            #if not success: break
           
        if args.simulation:
            Globals.sim.settle(animate=args.animation)
        
        if Globals.viewer and args.interactive:
            Globals.viewer.Idle()
            
        redprint("Demo %s Segment %s result: %s"%(demo_name, seg_name, success))
        
        if args.fake_data_demo and args.fake_data_segment and not args.simulation: break

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
