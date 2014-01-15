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
parser.add_argument("--cloud_proc_func", default="extract_red")
parser.add_argument("--cloud_proc_mod", default="hd_utils.cloud_proc_funcs")
    
parser.add_argument("--execution", type=int, default=0)
parser.add_argument("--animation", type=int, default=0)
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--cloud", type=int, default=0)
parser.add_argument("--downsample", help="downsample traj.", type=int, default=1)

parser.add_argument("--prompt", action="store_true")
parser.add_argument("--show_neighbors", action="store_true")
parser.add_argument("--select_manual", action="store_true")
parser.add_argument("--log", action="store_true")

parser.add_argument("--fake_data_demo", type=str)
parser.add_argument("--fake_data_segment",type=str)
parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx","ty","tz","rx","ry","rz"),
    default=[0,0,0,0,0,0], help="translation=(tx,ty,tz), axis-angle rotation=(rx,ry,rz)")


parser.add_argument("--trajopt_init",type=str,default="all_zero")
parser.add_argument("--pot_threshold",type=float, default=15)

parser.add_argument("--use_ar_init", action="store_true", default=False)
parser.add_argument("--ar_demo_file",type=str, default="")
parser.add_argument("--ar_run_file",type=str, default="")

parser.add_argument("--interactive",action="store_true")

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

import os, numpy as np, h5py, time, os.path as osp
import cPickle
import numpy as np
import importlib

import cloudprocpy, trajoptpy, openravepy

try:
    from hd_rapprentice import pr2_trajectories, PR2
    import rospy
except ImportError:
    print "Couldn't import ros stuff"


from hd_rapprentice import registration, animate_traj, ros2rave, \
     plotting_openrave, task_execution, \
     planning, tps, resampling
from hd_utils import yes_or_no, ros_utils as ru, func_utils, clouds, math_utils as mu
from hd_utils.pr2_utils import get_kinect_transform
from hd_utils.colorize import *
from hd_utils.utils import avg_transform
from hd_utils.defaults import demo_files_dir, hd_data_dir, asus_xtion_pro_f, \
        ar_init_dir, ar_init_demo_name, ar_init_playback_name, \
        tfm_head_dof, tfm_bf_head, tfm_gtf_ee
from hd_extract.extract_data import get_ar_marker_poses





cloud_proc_mod = importlib.import_module(args.cloud_proc_mod)
cloud_proc_func = getattr(cloud_proc_mod, args.cloud_proc_func)


class Globals:
    robot = None
    env = None

    pr2 = None


DS_SIZE = .025

def smaller_ang(x):
    return (x + np.pi)%(2*np.pi) - np.pi
def closer_ang(x,a,dir=0):
    """                                                
    find angle y (==x mod 2*pi) that is close to a                             
    dir == 0: minimize absolute value of difference                            
    dir == 1: y > x                                                            
    dir == 2: y < x                                                            
    """
    if dir == 0:
        return a + smaller_ang(x-a)
    elif dir == 1:
        return a + (x-a)%(2*np.pi)
    elif dir == -1:
        return a + (x-a)%(2*np.pi) - 2*np.pi


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
    
    
def set_gripper_maybesim(lr, value):
    if args.execution:
        gripper = {"l":Globals.pr2.lgrip, "r":Globals.pr2.rgrip}[lr]
        gripper.set_angle(value)
        Globals.pr2.join_all()
    else:
        Globals.robot.SetDOFValues([value*5], [Globals.robot.GetJoint("%s_gripper_l_finger_joint"%lr).GetDOFIndex()])
    return True


def exec_traj_maybesim(bodypart2traj):
    if args.animation:
        dof_inds = []
        trajs = []
        for (part_name, traj) in bodypart2traj.items():
            manip_name = {"larm":"leftarm","rarm":"rightarm"}[part_name]
            dof_inds.extend(Globals.robot.GetManipulator(manip_name).GetArmIndices())
            trajs.append(traj)
        full_traj = np.concatenate(trajs, axis=1)
        Globals.robot.SetActiveDOFs(dof_inds)
        animate_traj.animate_traj(full_traj, Globals.robot, restore=False,pause=True)
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
    for demo_name in demofile:
        for seg_name in demofile[demo_name]:
            if seg_name != 'done':
                keys[seg_num] = (demo_name, seg_name)
                print "%i: %s, %s"%(seg_num, demo_name, seg_name)
                seg_num += 1
            
    choice_ind = task_execution.request_int_in_range(seg_num)
    return keys[choice_ind]


def registration_cost(xyz0, xyz1):
    scaled_xyz0, _ = registration.unit_boxify(xyz0)
    scaled_xyz1, _ = registration.unit_boxify(xyz1)
    f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, rot_reg=1e-3, n_iter=30)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
    return cost


def find_closest_auto(demofile, new_xyz):
    if args.parallel:
        from joblib import Parallel, delayed
        
    demo_clouds = []
    
    
    keys = {}
    seg_num = 0
    for demo_name in demofile:
        for seg_name in demofile[demo_name]:
            keys[seg_num] = (demo_name, seg_name)
            seg_num += 1
            demo_clouds.append(np.asarray(demofile[demo_name][seg_name]["cloud_xyz"]))
            
    if args.parallel:
        costs = Parallel(n_jobs=3,verbose=100)(delayed(registration_cost)(demo_cloud, new_xyz) for demo_cloud in demo_clouds)
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
    return keys[ibest]

def find_closest_cloud(demofile, new_xyz):
    """
    recognition seems to be in john_python/lfd/recognition.py.
    """
    from hd_rapprentice import recognition
    demo_clouds = []
    
    
    keys = {}
    seg_num = 0
    for demo_name in demofile:
        for seg_name in demofile[demo_name]:
            keys[seg_num] = (demo_name, seg_name)
            seg_num += 1
            demo_clouds.append(np.asarray(demofile[demo_name][seg_name]["cloud_xyz"]))
    tps_func_vec = recognition.make_func_vec(new_xyz, clouds)
    costs = recognition.make_tps_dist_vec(tps_func_vec)
    ibest = np.argmin(costs)
    return keys[ibest]
    

def tpsrpm_plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f):
    ypred_nd = f.transform_points(x_nd)
    handles = []
    handles.append(Globals.env.plot3(ypred_nd, 3, (0,1,0)))
    #handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, x_nd.min(axis=0), x_nd.max(axis=0), xres = .1, yres = .1, zres = .04))
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

ADD_TABLE = True

def main():

    
    demotype_dir = osp.join(demo_files_dir, args.demo_type)
    h5file = osp.join(demotype_dir, args.demo_type+".h5")

    demofile = h5py.File(h5file, 'r')
    
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

    '''
    Add table
    '''
    # As found from measuring
    if args.execution:
        tablePos = [0.60,0.0,0.70]
    else:
        tablePos = [0.60,0.0,0.40]
    tableHalfExtents = [0.50,1.00,0.06]
    
    if ADD_TABLE:
        with Globals.env:
            body = openravepy.RaveCreateKinBody(Globals.env,'')
            body.SetName('table')
            body.InitFromBoxes(np.array([tablePos + tableHalfExtents]),True)
            Globals.env.AddKinBody(body,True)
        

    # get rgbd from pr2?
    if not args.fake_data_segment or not args.fake_data_demo:
        grabber = cloudprocpy.CloudGrabber()
        grabber.startRGBD()

    
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
        if not args.fake_data_segment or not args.fake_data_demo:
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
        if args.fake_data_segment and args.fake_data_demo:
            init_tfm = tfm_bf_head.dot(tfm_head_dof).dot(init_tfm)
        else:
            #T_w_k here should be different from rapprentice
            T_w_k = get_kinect_transform(Globals.robot)
            init_tfm = T_w_k.dot(init_tfm)

    
    
    Globals.viewer = trajoptpy.GetViewer(Globals.env)

    while True:
        '''
        Acquire point cloud
        '''
        redprint("Acquire point cloud")
        if args.fake_data_segment and args.fake_data_demo:
            
            #Set home position in sim
            l_vals = PR2.Arm.L_POSTURES['side']
            Globals.robot.SetDOFValues(l_vals, Globals.robot.GetManipulator('leftarm').GetArmIndices())
            Globals.robot.SetDOFValues(PR2.mirror_arm_joints(l_vals), Globals.robot.GetManipulator('rightarm').GetArmIndices())
            
            fake_seg = demofile[args.fake_data_demo][args.fake_data_segment]
            new_xyz = np.squeeze(fake_seg["cloud_xyz"])

            hmat = openravepy.matrixFromAxisAngle(args.fake_data_transform[3:6])
            hmat[:3,3] = args.fake_data_transform[0:3]
            if args.use_ar_init: hmat = init_tfm.dot(hmat)

            new_xyz = new_xyz.dot(hmat[:3,:3].T) + hmat[:3,3][None,:]
            #r2r = ros2rave.RosToRave(Globals.robot, np.asarray(fake_seg["joint_states"]["name"]))
            #r2r.set_values(Globals.robot, np.asarray(fake_seg["joint_states"]["position"][0]))
        else:
            Globals.pr2.head.set_pan_tilt(0,1.2)
            Globals.pr2.rarm.goto_posture('side')
            Globals.pr2.larm.goto_posture('side')
            Globals.pr2.join_all()
            time.sleep(.5)
        
            
            Globals.pr2.update_rave()
            
            rgb, depth = grabber.getRGBD()
            
            new_xyz = cloud_proc_func(rgb, depth, T_w_k)
        
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
        if args.select_manual:
            (demo_name, seg_name) = find_closest_manual(demofile, new_xyz)
        else:
            (demo_name, seg_name) = find_closest_auto(demofile, new_xyz)
        
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
            
        
        handles.append(Globals.env.plot3(old_xyz,5, (1,0,0)))
        handles.append(Globals.env.plot3(new_xyz,5, (0,0,1)))
        
        scaled_old_xyz, src_params = registration.unit_boxify(old_xyz)
        scaled_new_xyz, targ_params = registration.unit_boxify(new_xyz)
        f,_ = registration.tps_rpm_bij(scaled_old_xyz, scaled_new_xyz, plot_cb = tpsrpm_plot_cb,
                                       plotting=5 if args.animation else 0,rot_reg=np.r_[1e-4,1e-4,1e-1], n_iter=50, reg_init=10, reg_final=.1)
        f = registration.unscale_tps(f, src_params, targ_params)
        
        #handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, old_xyz.min(axis=0)-np.r_[0,0,.1], old_xyz.max(axis=0)+np.r_[0,0,.1], xres = .1, yres = .1, zres = .04))

#         import IPython
#         IPython.embed()

        eetraj = {}
        old_eetraj = {}
        for lr in 'lr':
            link_name = "%s_gripper_tool_frame"%lr
            
            old_ee_traj = np.asarray(seg_info[lr]["tfms_s"])
            #old_ee_traj = np.asarray(downsample_objects(seg_info[lr]["tfms_s"], args.downsample))
            
            
            if args.use_ar_init:
                for i in xrange(len(old_ee_traj)):
                    old_ee_traj[i] = init_tfm.dot(old_ee_traj[i])
            new_ee_traj = f.transform_hmats(np.asarray(old_ee_traj))

        
            eetraj[link_name] = new_ee_traj
#             eetraj[link_name] = np.asarray(old_ee_traj)
            
            handles.append(Globals.env.drawlinestrip(old_ee_traj[:,:3,3], 2, (1,0,0,1)))
            handles.append(Globals.env.drawlinestrip(new_ee_traj[:,:3,3], 2, (0,1,0,1)))
            
    
        '''
        Generating mini-trajectory
        '''
        miniseg_starts, miniseg_ends = split_trajectory_by_gripper(seg_info, args.pot_threshold)
        success = True
        redprint("mini segments: %s %s"%(miniseg_starts, miniseg_ends))
        
        prev_vals = {lr:None for lr in 'lr'}
        l_vals = PR2.Arm.L_POSTURES['side']
        for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):

            if args.execution=="real": Globals.pr2.update_rave()
            
            redprint("Generating joint trajectory for demo %s segment %s, part %i"%(demo_name, seg_name, i_miniseg))
            
            
            ### trajopt init traj
            init_joint_trajs = {}
            for lr in 'lr':
                link_name = "%s_gripper_tool_frame"%lr
                if args.trajopt_init == 'all_zero':
                    init_joint_traj = np.zeros((i_end+1-i_start, 7))
                    init_joint_trajs[lr] = init_joint_traj
                    
                elif args.trajopt_init == 'openrave_ik':
                    init_joint_traj = []
                    
                    manip_name = {"l":"leftarm", "r":"rightarm"}[lr]
                    manip = Globals.robot.GetManipulator(manip_name)
                    ik_type = openravepy.IkParameterizationType.Transform6D

                    all_x = []
                    x = []

                    for (i, pose_matrix) in enumerate(eetraj[link_name][i_start:i_end+1]):
                        
                        rot_pose_matrix = pose_matrix.dot(tfm_gtf_ee)
                        sol = manip.FindIKSolution(openravepy.IkParameterization(rot_pose_matrix, ik_type),
                                                   #openravepy.IkFilterOptions.IgnoreEndEffectorEnvCollisions,
                                                   openravepy.IkFilterOptions.CheckEnvCollisions)
                        all_x.append(i)
                        
                        if sol != None:
                            x.append(i)
                            init_joint_traj.append(sol)
                            blueprint("Openrave IK succeeds")
                        else:
                            redprint("Openrave IK fails")


                    if len(x) == 0:
                        if prev_vals[lr] is not None:
                            vals = prev_vals[lr]
                        else:
                            vals = l_vals if lr == 'l' else PR2.mirror_arm_joints(l_vals)
                        
                        init_joint_traj_interp = np.tile(vals,(i_end+1-i_start, 1))
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
                    
                    for (i, pose_matrix) in enumerate(eetraj[link_name][i_start:i_end+1]):
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
                    init_joint_traj = np.zeros((i_end+1-i_start, 7))
                    init_joint_trajs[lr] = init_joint_traj
                    
                
            ### Generate full-body trajectory
            bodypart2traj = {}
            for lr in 'lr':
                manip_name = {"l":"leftarm", "r":"rightarm"}[lr]

                ee_link_name = "%s_gripper_tool_frame"%lr
                new_ee_traj = eetraj[ee_link_name][i_start:i_end+1]
                
                if args.execution: Globals.pr2.update_rave()

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

                new_joint_traj = planning.plan_follow_traj(Globals.robot, manip_name,
                                                           Globals.robot.GetLink(ee_link_name),
                                                           new_ee_traj, init_joints     ,
                                                           end_pose_constraint=end_pose_constraint)
                
                prev_vals[lr] = new_joint_traj[-1]
                #handles.append(Globals.env.drawlinestrip(new_ee_traj[:,:3,3], 2, (0,1,0,1))
                
                
                part_name = {"l":"larm", "r":"rarm"}[lr]
                bodypart2traj[part_name] = new_joint_traj
                
                       
            redprint("Executing joint trajectory for demo %s segment %s, part %i using arms '%s'"%(demo_name, seg_name, i_miniseg, bodypart2traj.keys()))
            
            for lr in 'lr':
                success &= set_gripper_maybesim(lr, binarize_gripper(seg_info[lr]["pot_angles"][i_start], args.pot_threshold))
                # Doesn't actually check if grab occurred, unfortunately
                
            
            if not success: break
            
            if len(bodypart2traj) > 0:
                success &= exec_traj_maybesim(bodypart2traj)
                
            if not success: break
            
        redprint("Demo %s Segment %s result: %s"%(demo_name, seg_name, success))
        
        if args.fake_data_demo and args.fake_data_segment: break


if __name__ == "__main__":
    main()

