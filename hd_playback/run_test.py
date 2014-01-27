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


parser.add_argument("--init_state_h5", type=str)
parser.add_argument("--demo_name", type=str)
parser.add_argument("--perturb_name", type=str)

parser.add_argument("--demo_type", type=str)
parser.add_argument("--cloud_proc_func", default="extract_red")
parser.add_argument("--cloud_proc_mod", default="hd_utils.cloud_proc_funcs")
    
parser.add_argument("--execution", type=int, default=0)
parser.add_argument("--animation", type=int, default=0)
parser.add_argument("--simulation", type=int, default=0)
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--cloud", type=int, default=0)
parser.add_argument("--downsample", help="downsample traj.", type=int, default=1)

parser.add_argument("--prompt", action="store_true")
parser.add_argument("--show_neighbors", action="store_true")
parser.add_argument("--select", default="manual")
parser.add_argument("--log", action="store_true")


parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
    default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")


parser.add_argument("--trajopt_init",type=str,default="openrave_ik")
parser.add_argument("--pot_threshold",type=float, default=15)

parser.add_argument("--use_ar_init", action="store_true", default=False)
parser.add_argument("--ar_demo_file",type=str, default="")
parser.add_argument("--ar_run_file",type=str, default="")
parser.add_argument("--use_base", action="store_true", default=False)
parser.add_argument("--not_allow_base", help="dont allow base movement when use_base", action="store_true", default=False)
parser.add_argument("--early_stop_portion", help="stop early in the final segment to avoid bullet simulation problem", type=float, default=0.5)
parser.add_argument("--no_traj_resample", action="store_true", default=False)

parser.add_argument("--interactive",action="store_true")
parser.add_argument("--remove_table", action="store_true")

parser.add_argument("--friction", help="friction value in bullet", type=float, default=1.0)

parser.add_argument("--max_steps_before_failure", type=int, default=-1)
parser.add_argument("--tps_bend_cost_init", type=float, default=10)
parser.add_argument("--tps_bend_cost_final", type=float, default=.1)
parser.add_argument("--tps_n_iter", type=int, default=50)

parser.add_argument("--closest_rope_hack", action="store_true", default=False)
parser.add_argument("--closest_rope_hack_thresh", type=float, default=0.01)
parser.add_argument("--cloud_downsample", type=float, default=.01)



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

import os, numpy as np, h5py, time, os.path as osp
import cPickle
import numpy as np
import importlib
from numpy.linalg import norm


import cloudprocpy, trajoptpy, openravepy

try:
    from hd_rapprentice import pr2_trajectories, PR2
    import rospy
except ImportError:
    print "Couldn't import ros stuff"


from hd_rapprentice import registration, animate_traj, ros2rave, \
     plotting_openrave, task_execution, \
     planning, tps, resampling, \
     ropesim, rope_initialization
from hd_utils import yes_or_no, ros_utils as ru, func_utils, clouds, math_utils as mu, cloud_proc_funcs
from hd_utils.pr2_utils import get_kinect_transform
from hd_utils.colorize import *
from hd_utils.utils import avg_transform
from hd_utils.defaults import demo_files_dir, hd_data_dir, asus_xtion_pro_f, \
        ar_init_dir, ar_init_demo_name, ar_init_playback_name, \
        tfm_head_dof, tfm_bf_head, tfm_gtf_ee, cad_files_dir

from hd_extract.extract_data import get_ar_marker_poses




cloud_proc_mod = importlib.import_module(args.cloud_proc_mod)
cloud_proc_func = getattr(cloud_proc_mod, args.cloud_proc_func)
hitch_proc_func = getattr(cloud_proc_mod, "extract_hitch")


class Globals:
    robot = None
    env = None
    pr2 = None
    sim = None
    viewer = None


DS_SIZE = .025

def get_env_state():
    state =  [Globals.robot.GetTransform(), Globals.robot.GetDOFValues(), Globals.sim.rope.GetNodes()]
    hitch = Globals.env.GetKinBody('hitch')
    if hitch != None:
        state.append(hitch.GetTransform())
    return state
    

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


def split_trajectory_by_gripper(seg_info, pot_angle_threshold, thresh=5):
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
    
    new_seg_starts = []
    new_seg_ends = []
    for i in range(len(seg_starts)):
        if seg_ends[i]- seg_starts[i] >= thresh:
            new_seg_starts.append(seg_starts[i])
            new_seg_ends.append(seg_ends[i])
    
#     import IPython
#     IPython.embed()

    return new_seg_starts, new_seg_ends

def binarize_gripper(angle, pot_angle_threshold):
    open_angle = .08
    closed_angle = 0
    if angle > pot_angle_threshold: return open_angle
    else: return closed_angle
    
    
def set_gripper_maybesim(lr, is_open, prev_is_open):
    mult = 1 if args.execution else 5
    open_angle = .08 * mult
    closed_angle = (0 if not args.simulation else .02) * mult
    
    target_val = open_angle if is_open else closed_angle

    if args.execution:
        gripper = {"l":Globals.pr2.lgrip, "r":Globals.pr2.rgrip}[lr]
        gripper.set_angle(target_val)
        Globals.pr2.join_all()
        
    elif not args.simulation:
        Globals.robot.SetDOFValues([target_val], [Globals.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()])
        
    elif args.simulation:
        # release constraints if necessary
        if is_open and not prev_is_open:
            Globals.sim.release_rope(lr)

        # execute gripper open/close trajectory
        joint_ind = Globals.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()
        start_val = Globals.robot.GetDOFValues([joint_ind])[0]
        joint_traj = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / .02))
        for val in joint_traj:
            Globals.robot.SetDOFValues([val], [joint_ind])
            Globals.sim.step()
            if args.animation:
                Globals.viewer.Step()
                if args.interactive: Globals.viewer.Idle()
        # add constraints if necessary
        if not is_open and prev_is_open:
            
            if not Globals.sim.grab_rope(lr):
                redprint("Grab failed")
                return False
            else:
                blueprint("Grab succeeded")
        
        
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

def exec_traj_maybesim(bodypart2traj):
    def sim_callback(i):
        Globals.sim.step()
    
    if args.animation or args.simulation:
        dof_inds = []
        trajs = []
        base_hmats = None
        for (part_name, traj) in bodypart2traj.items():
            if part_name == "base":
                base_hmats = [planning.base_pose_to_mat(base_dofs) for base_dofs in traj]
                continue
            manip_name = {"larm":"leftarm","rarm":"rightarm"}[part_name]
            dof_inds.extend(Globals.robot.GetManipulator(manip_name).GetArmIndices())
            trajs.append(traj)
        full_traj = np.concatenate(trajs, axis=1)
        Globals.robot.SetActiveDOFs(dof_inds)
        
        if args.simulation:
            # make the trajectory slow enough for the simulation
            full_traj, base_hmats = ropesim.retime_traj(Globals.robot, dof_inds, full_traj, base_hmats)
            
            # in simulation mode, we must make sure to gradually move to the new starting position
            curr_vals       = Globals.robot.GetActiveDOFValues()
            transition_traj = np.r_[[curr_vals], [full_traj[0]]]

            if base_hmats != None:
                transition_base_hmats = [Globals.robot.GetTransform()] + [base_hmats[0]]
            else:
                transition_base_hmats = None
            
            unwrap_in_place(transition_traj)
            
            transition_traj, transition_base_hmats = ropesim.retime_traj(Globals.robot, dof_inds, transition_traj, transition_base_hmats, max_cart_vel=.01)
            animate_traj.animate_traj(transition_traj, transition_base_hmats, Globals.robot, restore=False, pause=args.interactive,
                callback=sim_callback if args.simulation else None, step_viewer=args.animation)
            
            full_traj[0] = transition_traj[-1]
          
            
            if base_hmats != None:
                base_hmats[0] = transition_base_hmats[-1]
            
            
            unwrap_in_place(full_traj)
        
        animate_traj.animate_traj(full_traj, base_hmats, Globals.robot, restore=False, pause=args.interactive,
                                  callback=sim_callback if args.simulation else None, step_viewer=args.animation)
        
        
    if args.execution:
        if not args.prompt or yes_or_no("execute?"):
            pr2_trajectories.follow_body_traj(Globals.pr2, bodypart2traj)
        else:
            return False

    return True

def find_closest_manual(demofile, _new_xyz):
    """for now, just prompt the user"""
    

    print "choose from the following options (type an integer)"
    seg_num = 0
    keys = {}
    is_finalsegs = {}
    for demo_name in demofile:
        if demo_name != "ar_demo":
            if 'done' in demofile[demo_name].keys():
                final_seg_id = len(demofile[demo_name].keys()) - 2
            else:
                final_seg_id = len(demofile[demo_name].keys()) - 1
                
            
            for seg_name in demofile[demo_name]:
                if seg_name != 'done':
                    keys[seg_num] = (demo_name, seg_name)
                    print "%i: %s, %s"%(seg_num, demo_name, seg_name)
                    
                    if seg_name == "seg%02d"%(final_seg_id):
                        is_finalsegs[seg_num] = True
                    else:
                        is_finalsegs[seg_num] = False

                    seg_num += 1

    choice_ind = task_execution.request_int_in_range(seg_num)
    return keys[choice_ind], is_finalsegs[choice_ind]


def registration_cost(xyz0, xyz1):
    scaled_xyz0, _ = registration.unit_boxify(xyz0)
    scaled_xyz1, _ = registration.unit_boxify(xyz1)
    f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, rot_reg=1e-3, n_iter=30)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    return cost


def find_closest_auto(demofile, new_xyz, init_tfm=None, n_jobs=3):
    if args.parallel:
        from joblib import Parallel, delayed
        
    demo_clouds = []
    
    DS_LEAF_SIZE = 0.045
    new_xyz = clouds.downsample(new_xyz,DS_LEAF_SIZE)
    
    avg = 0.0
    
    keys = {}
    is_finalsegs = {}

    seg_num = 0
    for demo_name in demofile:
        if demo_name != "ar_demo":
            if 'done' in demofile[demo_name].keys():
                final_seg_id = len(demofile[demo_name].keys()) - 2
            else:
                final_seg_id = len(demofile[demo_name].keys()) - 1

            for seg_name in demofile[demo_name]:
                keys[seg_num] = (demo_name, seg_name)
                
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
    
    ibest = np.argmin(costs)
    return keys[ibest], is_finalsegs[ibest]


def find_closest_clusters(demofile, clusterfile, new_xyz, init_tfm=None, check_n=2, n_jobs=3):
    if args.parallel:
        from joblib import Parallel, delayed
    
    DS_LEAF_SIZE = 0.045
    new_xyz = clouds.downsample(new_xyz,DS_LEAF_SIZE)
        
    # Store all the best cluster clouds
    cluster_clouds = {}
    keys = clusterfile['keys']
    clusters = clusterfile['clusters']
    for cluster in clusters:
        best_seg = clusters[cluster][0]
        dname, sname = keys[best_seg]
        cloud = clouds.downsample(np.asarray(demofile[dname][sname]["cloud_xyz"]),DS_LEAF_SIZE)
        if init_tfm is not None:
            cloud = cloud.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]

        cluster_clouds[cluster] = cloud

    # Check the clusters with min costs
    if args.parallel:
        ccosts = Parallel(n_jobs=n_jobs,verbose=51)(delayed(registration_cost)(cluster_clouds[i], new_xyz) for i in cluster_clouds)
    else:
        ccosts = []
        for (i,ds_cloud) in cluster_clouds.items():
            ccosts.append(registration_cost(ds_cloud, new_xyz))
            print "completed %i/%i"%(i+1, len(cluster_clouds))
    
    print "Cluster costs: \n", ccosts

    
    best_clusters = np.argsort(ccosts)
    check_n = min(check_n, len(best_clusters))

    is_finalsegs = {}    
    check_clouds = {}
    best_segs = []
    for c in best_clusters[:check_n]:
        cluster_segs = clusters[c]
        best_segs.extend(cluster_segs)
        for seg in cluster_segs:
            dname,sname = keys[seg]
            check_clouds[seg] = clouds.downsample(np.asarray(demofile[dname][sname]["cloud_xyz"]),DS_LEAF_SIZE)
            if 'done' in demofile[dname].keys():
                final_seg_id = len(demofile[dname].keys()) - 2
            else:
                final_seg_id = len(demofile[dname].keys()) - 1

            if sname == "seg%02d"%(final_seg_id):
                is_finalsegs[seg] = True
            else:
                is_finalsegs[seg] = False

    # Check the clusters with min costs
    if args.parallel:
        costs = Parallel(n_jobs=n_jobs,verbose=51)(delayed(registration_cost)(check_clouds[i], new_xyz) for i in check_clouds)
    else:
        costs = []
        for (i,ds_cloud) in check_clouds.items():
            costs.append(registration_cost(ds_cloud, new_xyz))
            print "completed %i/%i"%(i+1, len(check_clouds))
    
    print "Costs: \n", costs
    
    if args.show_neighbors:
        nshow = min(5, len(check_clouds))
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
    
    ibest = best_segs[np.argmin(costs)]
    
    return keys[ibest], is_finalsegs[ibest]



def tpsrpm_plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f):
    ypred_nd = f.transform_points(x_nd)
    handles = []
    handles.append(Globals.env.plot3(ypred_nd, 3, (0,1,0)))
    #handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, x_nd.min(axis=0), x_nd.max(axis=0), xres = .1, yres = .1, zres = .04))
    if Globals.viewer:
        Globals.viewer.Step()
    
def arm_moved(joint_traj):
    if len(joint_traj) < 2: return False
    return ((joint_traj[1:] - joint_traj[:-1]).ptp(axis=0) > .01).any()
    
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


def main():
    init_state_h5file = h5py.File(args.init_state_h5+".h5", "r")
    print args.init_state_h5+".h5"
    
    demotype_dir = osp.join(demo_files_dir, args.demo_type)
    demo_h5file = osp.join(demotype_dir, args.demo_type+".h5")
    print demo_h5file
    demofile = h5py.File(demo_h5file, 'r')
    
    if args.select == "clusters":
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
    if args.execution:
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
                
                pc = ru.xyzrgb2pc(xyz, rgb)
                
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
        if args.simulation:
            init_tfm = tfm_bf_head.dot(tfm_head_dof).dot(init_tfm)
        else:
            #T_w_k here should be different from rapprentice
            T_w_k = get_kinect_transform(Globals.robot)
            init_tfm = T_w_k.dot(init_tfm)

    if args.simulation:

        if has_hitch(init_state_h5file, args.demo_name, args.perturb_name):
            Globals.env.Load(osp.join(cad_files_dir, 'hitch.xml'))
            hitch_pos = init_state_h5file[args.demo_name][args.perturb_name]['hitch_pos']
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

    seg_env_state = []

    while True:
        
        if args.max_steps_before_failure != -1 and curr_step > args.max_steps_before_failure:
            redprint("Number of steps %d exceeded maximum %d" % (curr_step, args.max_steps_before_failure))
            break

        seg_state = []
        curr_step += 1
        '''
        Acquire point cloud
        '''
        redprint("Acquire point cloud")
        
        
        rope_cloud = None     
                   
        if args.simulation:
            
            #Set home position in sim
            l_vals = PR2.Arm.L_POSTURES['side']
            Globals.robot.SetDOFValues(l_vals, Globals.robot.GetManipulator('leftarm').GetArmIndices())
            Globals.robot.SetDOFValues(PR2.mirror_arm_joints(l_vals), Globals.robot.GetManipulator('rightarm').GetArmIndices())

            if args.simulation and curr_step > 1:
                # for following steps in rope simulation, using simulation result
                new_xyz = Globals.sim.observe_cloud(3)
                new_xyz = clouds.downsample(new_xyz, args.cloud_downsample)
                
                hitch = Globals.env.GetKinBody('hitch')
                                
                if hitch != None:
                    pos = hitch.GetTransform()[:3,3]
                    hitch_height = hitch_body.GetLinks()[0].GetGeometries()[0].GetCylinderHeight()
                    pos[2] = pos[2] - hitch_height/2
                    hitch_cloud = cloud_proc_funcs.generate_hitch_points(pos)
                    hitch_cloud = clouds.downsample(hitch_cloud, args.cloud_downsample)
                    new_xyz = np.r_[new_xyz, hitch_cloud]
                
            else:          
                init_seg = init_state_h5file[args.demo_name][args.perturb_name]
                new_xyz = np.squeeze(init_seg["cloud_xyz"])
        
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
                    
                    hitch = Globals.env.GetKinBody('hitch')
                    if hitch != None:
                        pos = hitch.GetTransform()[:3,3]
                        hitch_height = hitch_body.GetLinks()[0].GetGeometries()[0].GetCylinderHeight()
                        pos[2] = pos[2] - hitch_height/2
                        hitch_cloud = cloud_proc_funcs.generate_hitch_points(pos)
                        hitch_cloud = clouds.downsample(hitch_cloud, args.cloud_downsample)
                        new_xyz = np.r_[new_xyz, hitch_cloud]
            
            if args.closest_rope_hack:
                if args.simulation:
                    rope_cloud = Globals.sim.observe_cloud(3)
                    rope_cloud = clouds.downsample(rope_cloud, args.cloud_downsample)
                else:
                    if has_hitch(init_state_h5file, args.demo_name, args.perturb_name):
                        rope_cloud = demofile[args.demo_name][args.perturb_name]['object']
                    else:
                        rope_cloud = demofile[args.demo_name][args.perturb_name]['cloud_xyz']


        else:
            Globals.pr2.head.set_pan_tilt(0,1.2)
            Globals.pr2.rarm.goto_posture('side')
            Globals.pr2.larm.goto_posture('side')
            Globals.pr2.join_all()
            time.sleep(3.5)
        
            Globals.pr2.update_rave()
            
            rgb, depth = grabber.getRGBD()
            
            new_xyz = cloud_proc_func(rgb, depth, T_w_k)
            new_xyz = clouds.downsample(new_xyz, args.cloud_downsample)
            
            if args.closest_rope_hack:
                rope_cloud = np.array(new_xyz)
            
            if has_hitch(demofile):
                hitch_normal = clouds.clouds_plane(new_xyz)
                
                hitch_cloud, hitch_pos = hitch_proc_func(rgb, depth, T_w_k, hitch_normal)
                hitch_cloud = clouds.downsample(hitch_cloud, args.cloud_downsample)
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
        if args.select=="manual":
            (demo_name, seg_name), is_final_seg = find_closest_manual(demofile, new_xyz)
        elif args.select=="auto":
            (demo_name, seg_name), is_final_seg = find_closest_auto(demofile, new_xyz, init_tfm)
        else:
            (demo_name, seg_name), is_final_seg = find_closest_clusters(demofile, clusterfile, new_xyz, init_tfm)

        
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
        miniseg_starts, miniseg_ends = split_trajectory_by_gripper(seg_info, args.pot_threshold)
        success = True
        redprint("mini segments: %s %s"%(miniseg_starts, miniseg_ends))
        
        segment_len = miniseg_ends[-1] - miniseg_starts[0] + 1
        portion = max(args.early_stop_portion, miniseg_ends[0] / float(segment_len))
        
        prev_vals = {lr:None for lr in 'lr'}
        l_vals = PR2.Arm.L_POSTURES['side']
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
                                    reference_sol = l_vals if lr == 'l' else PR2.mirror_arm_joints(l_vals)
                        
                            
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
                            vals = l_vals if lr == 'l' else PR2.mirror_arm_joints(l_vals)
                        
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
                gripper_open = binarize_gripper(seg_info[lr]["pot_angles"][i_start], args.pot_threshold)
                prev_gripper_open = binarize_gripper(seg_info[lr]["pot_angles"][i_start-1], args.pot_threshold) if i_start != 0 else False
                if not set_gripper_maybesim(lr, gripper_open, prev_gripper_open):
                    redprint("Grab %s failed"%lr)
                    success = False
                # Doesn't actually check if grab occurred, unfortunately

            seg_state.append(get_env_state())            
            # if not success: break
            
            if len(bodypart2traj['larm']) > 0:
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
                if args.execution:
                    time.sleep(5)
            
            seg_env_state.append(seg_state) 

            #if not success: break
            
            
        if args.simulation:
            Globals.sim.settle(animate=args.animation)

        if Globals.viewer and args.interactive:
            Globals.viewer.Idle()
            
        redprint("Demo %s Segment %s result: %s"%(demo_name, seg_name, success))

    init_state_h5file.close()
    demofile.close()
        

if __name__ == "__main__":
    main()

