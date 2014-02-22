#!/usr/bin/env python
###################
"""
Workflow:
1. Fake data + animation only
    --fake_data_segment=xxx --execution=0
2. Fake data + Gazebo. Set Gazebo to initial state of fake data segment so we'll execute the same thing.
    --fake_data_segment=xxx --execution=1
    This is just so we know the robot won't do something stupid that we didn't catch with openrave only mode.
3. Real data + Gazebo
    --execution=1
    The problem is that the gazebo robot is in a different state from the real robot, in particular, the head tilt
    angle. TODO: write a script that       sets gazebo head to real robot head
4. Real data + Real execution.
    --execution=1

The question is, do you update the robot's head transform.
If you're using fake data, don't update it.

"""
from rapprentice import registration, colorize, \
    animate_traj, ros2rave, plotting_openrave, task_execution, \
    planning, func_utils, resampling, rope_initialization, clouds, ropesim_floating
from rapprentice import math_utils as mu
from knot_classifier import isKnot as is_knot

import trajoptpy
import openravepy
import numpy as np
import h5py
from numpy import asarray
import atexit
import time
import sys
import random
import copy
import dhm_utils as dhm_u
import IPython as ipy

#Don't use args, use globals
#args = None


L_POSTURES = {'side': np.array([[-0.98108876, -0.1846131 ,  0.0581623 ,  0.10118172],
                                [-0.19076337,  0.97311662, -0.12904799,  0.68224057],
                                [-0.03277475, -0.13770277, -0.98993119,  0.91652485],
                                [ 0.        ,  0.        ,  0.        ,  1.        ]]) }

R_POSTURES = {'side' : np.array([[-0.98108876,  0.1846131 ,  0.0581623 ,  0.10118172],
                                 [ 0.19076337,  0.97311662,  0.12904799, -0.68224057],
                                 [-0.03277475,  0.13770277, -0.98993119,  0.91652485],
                                 [ 0.        ,  0.        ,  0.        ,  1.        ]]) }


def move_sim_arms_to_side():
    """Moves the simulated arms to the side."""
    Globals.sim.grippers['r'].set_toolframe_transform(R_POSTURES['side'])
    Globals.sim.grippers['l'].set_toolframe_transform(L_POSTURES['side'])
    

class Globals:
    env = None
    sim = None
    exec_log = None
    viewer = None
    random_seed = None


class RopeState:
    def __init__(self, segment, perturb_radius, perturb_num_points):
        self.segment = segment
        self.perturb_radius = perturb_radius
        self.perturb_num_points = perturb_num_points

class TaskParameters:
    def __init__(self, action_file, cloud_xyz, animate=False, warp_root=True, max_steps_before_failure=5,
                 no_cmat=False):
        self.action_file = action_file
        self.cloud_xyz   = cloud_xyz
        self.animate     = animate
        self.warp_root   = warp_root
        self.max_steps_before_failure = max_steps_before_failure
        self.no_cmat=no_cmat


#init_rope_state_segment, perturb_radius, perturb_num_points
def redprint(msg):
    """
    Print the message to the console in red, bold font.
    """
    print colorize.colorize(msg, "red", bold=True)

def yellowprint(msg):
    """
    Print the message to the console in red, bold font.
    """
    print colorize.colorize(msg, "yellow", bold=True)

def split_trajectory_by_gripper(rgrip_joints, lgrip_joints):
    """
    Split up the trajectory into sections with breaks occuring when the grippers open or close.
    Return: (seg_starts, seg_ends)
    """
    assert len(rgrip_joints)==len(lgrip_joints), "joint trajectory length mis-match."

    thresh = .04  # open/close threshold
    n_steps = len(lgrip_joints)

    # indices BEFORE transition occurs
    l_openings = np.flatnonzero((lgrip_joints[1:] >= thresh) & (lgrip_joints[:-1] < thresh))
    r_openings = np.flatnonzero((rgrip_joints[1:] >= thresh) & (rgrip_joints[:-1] < thresh))
    l_closings = np.flatnonzero((lgrip_joints[1:] < thresh) & (lgrip_joints[:-1] >= thresh))
    r_closings = np.flatnonzero((rgrip_joints[1:] < thresh) & (rgrip_joints[:-1] >= thresh))

    before_transitions = np.r_[l_openings, r_openings, l_closings, r_closings]
    after_transitions  = before_transitions + 1
    seg_starts         = np.unique(np.r_[0, after_transitions])
    seg_ends           = np.unique(np.r_[before_transitions, n_steps - 1])

    return seg_starts, seg_ends


def binarize_gripper(angle):
    thresh = .04
    return angle > thresh


def load_random_start_segment(demofile):
    start_keys = [k for k in demofile.keys() if k.startswith('demo') and k.endswith('00')]
    seg_name = random.choice(start_keys)
    return demofile[seg_name]['cloud_xyz']

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



def sample_rope_state(demofile, perturb_points=5, min_rad=0, max_rad=.15, rotation=False):
    """
    samples a rope state, by picking a random segment, perturbing, rotating about the median, 
    then setting a random translation such that the rope is essentially within grasp room
    """

    # TODO: pick a random rope initialization
    new_xyz= load_random_start_segment(demofile)
    perturb_radius = random.uniform(min_rad, max_rad)
    rope_nodes = rope_initialization.find_path_through_point_cloud( new_xyz,
                                                                    perturb_peak_dist=perturb_radius,
                                                                    num_perturb_points=perturb_points)
    if rotation:
        rand_theta = rotation*np.random.rand()
        rope_nodes = rotate_about_median(rope_nodes, rand_theta)
        rope_nodes = place_in_feasible_region(rope_nodes)
    return rope_nodes

def set_gripper_sim(lr, is_open, prev_is_open, animate=True):
    """Opens or closes the gripper. Also steps the simulation.

    Arguments:
        is_open -- boolean that is true if the gripper should be open
        prev_is_open -- boolean that is true if the gripper was open last step
    May send an open command if is_open is true but prev_is_open is false.

    Return False if the simulated gripper failed to grab the rope, eles return True.
    """
    mult         = 5
    open_angle   = .08 * mult
    closed_angle = .02 * mult

    target_val = open_angle if is_open else closed_angle

    # release constraints if necessary
    if is_open and not prev_is_open:
        Globals.sim.release_rope(lr)

    # execute gripper open/close trajectory
    start_val = Globals.sim.grippers[lr].get_gripper_joint_value()

    #gripper_velocity = 0.2
    #  a smaller number makes the gripper move slower.
    #  if the gripper moves slower then it will fling the rope less.
    gripper_velocity = 0.005  ##<<=== is this too small??
    joint_traj       = np.linspace(start_val, target_val, np.ceil(abs(target_val - start_val) / gripper_velocity))
    #print "joint_traj after retime =", joint_traj
    for i, val in enumerate(joint_traj):
        Globals.sim.grippers[lr].set_gripper_joint_value(val)
        Globals.sim.step()
        if animate and not i%10:
            Globals.viewer.Step()

    if not is_open and prev_is_open:
        if not Globals.sim.grab_rope(lr):
            return False
    return True


def unwrap_arm_traj_in_place(traj):
    assert traj.shape[1] == 7
    for i in [2, 4, 6]:
        traj[:, i] = np.unwrap(traj[:, i])
    return traj


def unwrap_in_place(t):
    # TODO: do something smarter than just checking shape[1]
    if t.shape[1] == 7:
        unwrap_arm_traj_in_place(t)
    elif t.shape[1] == 14:
        unwrap_arm_traj_in_place(t[:, :7])
        unwrap_arm_traj_in_place(t[:, 7:])
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



#TODO: possibly memoize
#@func_utils.once
def get_downsampled_clouds(values, DS_SIZE):
    cloud_list = []
    shapes = []
    for seg in values:
        cloud = seg["cloud_xyz"]
        cloud = cloud[...].copy()
        if cloud.shape[0] > 200:
            down_cloud = clouds.downsample(cloud, DS_SIZE)
        else:
            down_cloud = clouds.downsample(cloud, DS_SIZE)

        cloud_list.append(down_cloud)
        shapes.append(down_cloud.shape)
    return cloud_list, shapes


def remove_inds(a, inds):
    return [x for (i, x) in enumerate(a) if i not in inds]


def find_closest_manual(demofile, _new_xyz):
    """for now, just prompt the user"""
    seg_names = demofile.keys()
    print_string = "choose from the following options (type an integer). Enter a negative number to exit."
    print print_string
    for (i, seg_name) in enumerate(seg_names):
        print "%i: %s" % (i, seg_name)
    print print_string
    choice_ind = task_execution.request_int_in_range(len(seg_names))
    if choice_ind < 0:
        return None
    chosen_seg = seg_names[choice_ind]
    return chosen_seg


def registration_cost(xyz0, xyz1):
    scaled_xyz0, _ = registration.unit_boxify(xyz0)
    scaled_xyz1, _ = registration.unit_boxify(xyz1)
    
    #import matplotlib.pylab as plt
    #plt.scatter(scaled_xyz0[:,0], scaled_xyz0[:,1], c='r' )
    #plt.hold(True)
    #plt.scatter(scaled_xyz1[:,0], scaled_xyz1[:,1], c='b' )
    #plt.show()

    #print xyz0.shape, xyz1.shape
    f, g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, n_iter=10, rot_reg=1e-3)
    cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)   
    return cost


def auto_choose(actionfile, new_xyz, nparallel=-1):
    """
    @param demofile: h5py.File object
    @param new_xyz : new rope point-cloud
    @nparallel     : number of parallel jobs to run for tps cost calculaion.
                     If -1 only 1 job is used (no parallelization).
    
    @return          : return the name of the segment with the lowest warping cost.
    """
    if not nparallel == -1:
        from joblib import Parallel, delayed
        nparallel = min(nparallel, 8)

    demo_data = actionfile.items()

    if nparallel != -1:
        before = time.time()
        redprint("auto choose parallel with njobs = %d"%nparallel)
        costs  = Parallel(n_jobs=nparallel, verbose=0)(delayed(registration_cost)(ddata[1]['cloud_xyz'][:], new_xyz) for ddata in demo_data)
        after  = time.time()
        print "Parallel registration time in seconds =", after - before
    else:
        costs = []
        redprint("auto choose sequential..")
        for i, ddata in enumerate(demo_data):
            costs.append(registration_cost(ddata[1]['cloud_xyz'][:], new_xyz))
            print(("tps-cost completed %i/%i" % (i + 1, len(demo_data))))

    ibest = np.argmin(costs)
    redprint ("auto choose returning..")
    return demo_data[ibest][0]



def arm_moved(hmat_traj):
    return True
    if len(hmat_traj) < 2:
        return False
    tts = hmat_traj[:,:3,3]
    return ((tts[1:] - tts[:-1]).ptp(axis=0) > .01).any()


def tpsrpm_plot_cb(x_nd, y_md, targ_Nd, corr_nm, wt_n, f, old_xyz, new_xyz, last_one=False):
    _, src_params = registration.unit_boxify(old_xyz)
    _, targ_params = registration.unit_boxify(new_xyz)
    f = registration.unscale_tps(f, src_params, targ_params)
    #ypred_nd = f.transform_points(x_nd)
    handles = []
    #handles.append(Globals.env.plot3(ypred_nd, 3, (0, 1, 0, 1)))
    ypred_nd = f.transform_points(old_xyz)
    handles.append(Globals.env.plot3(ypred_nd, 3, (0, 1, 0, 1)))
    handles.extend(plotting_openrave.draw_grid(Globals.env, f.transform_points, old_xyz.min(axis=0) - np.r_[0, 0, .1],
                                               old_xyz.max(axis=0) + np.r_[0, 0, .1], xres=.1, yres=.1, zres=.04))

    if Globals.viewer:
        Globals.viewer.Step()
        time.sleep(0.1)
        #Globals.viewer.Idle()


def load_segment(demofile, segment, fake_data_transform=[0, 0, 0, 0, 0, 0]):
    fake_seg = demofile[segment]
    new_xyz = np.squeeze(fake_seg["cloud_xyz"])
    hmat = openravepy.matrixFromAxisAngle(fake_data_transform[3:6])  # @UndefinedVariable
    hmat[:3, 3] = fake_data_transform[0:3]
    new_xyz = new_xyz.dot(hmat[:3, :3].T) + hmat[:3, 3][None, :]
    r2r = ros2rave.RosToRave(Globals.robot, asarray(fake_seg["joint_states"]["name"]))
    return new_xyz, r2r


def unif_resample(traj, max_diff, wt=None):
    """
    Resample a trajectory so steps have same length in joint space
    """
    import scipy.interpolate as si

    tol = .005
    if wt is not None:
        wt = np.atleast_2d(wt)
        traj = traj * wt

    dl = mu.norms(traj[1:] - traj[:-1], 1)
    l = np.cumsum(np.r_[0, dl])
    goodinds = np.r_[True, dl > 1e-8]
    deg = min(3, sum(goodinds) - 1)
    if deg < 1:
        return traj, np.arange(len(traj))

    nsteps = max(int(np.ceil(float(l[-1]) / max_diff)), 2)
    newl = np.linspace(0, l[-1], nsteps)

    ncols = traj.shape[1]
    colstep = 10
    traj_rs = np.empty((nsteps, ncols))
    for istart in xrange(0, traj.shape[1], colstep):
        (tck, _) = si.splprep(traj[goodinds, istart:istart + colstep].T, k=deg, s=tol ** 2 * len(traj), u=l[goodinds])
        traj_rs[:, istart:istart + colstep] = np.array(si.splev(newl, tck)).T
    if wt is not None:
        traj_rs = traj_rs / wt

    newt = np.interp(newl, l, np.arange(len(traj)))
    return traj_rs, newt


def make_table_xml(translation, extents):
    xml = """
    <Environment>
      <KinBody name="table">
        <Body type="static" name="table_link">
          <Geom type="box">
            <Translation>%f %f %f</Translation>
            <extents>%f %f %f</extents>
            <diffuseColor>.96 .87 .70</diffuseColor>
          </Geom>
        </Body>
      </KinBody>
    </Environment>
    """ % (translation[0], translation[1], translation[2], extents[0], extents[1], extents[2])
    return xml



def do_single_task(task_params):
    """
    Do one task.

    Arguments:
    task_params -- a task_params object
    task_params.action_file : h5 file for demonstration. It is the filename. This has the
                              bootstrap trees.
    task_params.start_state : the initial point_cloud
    task_params.animate : boolean
    If task_parms.max_steps_before failure is -1, then it loops until the knot is detected.

    Return {'success': <boolean>, 'seg_info':[{ ...}, {...}, ...]}
    seg_info hashes are of the form
    {'parent': The name of the segment from the action file that was chosen, 'hmats': trajectories for the left and
    right grippers, 'cloud_xyz': the new pointcloud, 'cmat': the correspondence matrix or list of segments}
    """
    #Begin: setup local variables from parameters
    action_file   = task_params.action_file
    animate       = task_params.animate
    max_steps_before_failure = task_params.max_steps_before_failure
    choose_segment = auto_choose
    knot = "any"
    no_cmat = task_params.no_cmat

    ### Setup ###
    demofile = setup_and_return_action_file(action_file, task_params.cloud_xyz, animate=animate)

    knot_results = []
    loop_results = []
    i = 0
    move_sim_arms_to_side()

    while True:
        print "max_steps_before_failure =", max_steps_before_failure
        print "i =", i
        if max_steps_before_failure != -1 and i >= max_steps_before_failure:
            break
        loop_result = loop_body(task_params, demofile, choose_segment, knot, animate, curr_step=i, no_cmat=no_cmat)
        if loop_result is not None:
            knot_result = loop_result['found_knot']
            loop_results.append(loop_result)
        else:
            knot_result = None
        redprint('knot results:\t' + str(knot_result))
        knot_results.append(knot_result)
        #Break if it either sucessfully ties a knot (knot_result is True), or the main loop wants to exit (knot_result is None)
        if knot_result or knot_result==None:
            break
        i += 1
    demofile.close()
    seg_info_list = [loop_result['seg_info'] for loop_result in loop_results]
    return {'success':knot_results[-1], 'seg_info':seg_info_list}


def add_loop_results_to_hdf5(demofile, loop_results):
    """Saves the loop_results in the hdf5 file demofile.
    Arguments: demofile is the h5py handle to the already open hdf5 file.
    loop_results is a list of dicts: [{'found_knot': found_knot, 'segment': segment, 'link2eetraj': link2eetraj,
    'new_xyz': new_xyz} ... ]
    """
    #TODO: unit test this function
    #TODO: also append a random number for all the segments (ie. generate one before the for loop)
    for loop_result in loop_results:
        parent_name = loop_result['segment']
        parent = demofile[parent_name]
        child_name = '/' + parent_name + '_' + str(random.randint(0, 10000))
        #Make a copy of the parent
        #TODO: figure out which args are necessary
        parent.copy(parent, child_name, shallow=False, expand_soft=True, expand_external=True, expand_refs=True)
        #parent.copy(parent, child_name)
        child = demofile[child_name]
        #Now update the child with loop_result
        for lr in 'lr':
            #TODO: test this step in ipython
            link_name = "%s_gripper_tool_frame" % lr
            child[link_name]["hmat"][...] = loop_result['link2eetraj'][link_name]
        del child["cloud_xyz"]
        child["cloud_xyz"] = loop_result['new_xyz']
        demofile.flush()
        if not "derived" in child.keys():
            child["derived"] = True
        demofile.flush()


def set_random_seed(task_params):
    if task_params.random_seed:
        Globals.random_seed = task_params.random_seed
        print "Found a random seed"
        print "Random seed is", Globals.random_seed


def setup_log(filename):
    if filename:
        if Globals.exec_log is None:
            redprint("Writing log to file %s" % filename)
            Globals.exec_log = task_execution.ExecutionLog(filename)
            #This will flush to the log when the program closes.
            atexit.register(Globals.exec_log.close)

#TODO  Consider encapsulating these intermedite return values in a class.
def setup_and_return_action_file(action_file, new_xyz, animate):
    """For the simulation, this code runs before the main loop. It also sets the numpy random seed"""
    if Globals.random_seed is not None:
        np.random.seed(Globals.random_seed)

    demofile    = h5py.File(action_file, 'r+')
    Globals.env = openravepy.Environment()  # @UndefinedVariable
    
    table_height = new_xyz[:, 2].mean() - 0.17
    table_xml    = make_table_xml(translation=[1, 0, table_height], extents=[1, 0.6, .01])
    Globals.env.LoadData(table_xml)
    
    Globals.sim = ropesim_floating.FloatingGripperSimulation(Globals.env)
    move_sim_arms_to_side()
    
    Globals.sim.create(new_xyz)
    Globals.sim.settle()    
    if animate:
        Globals.viewer = trajoptpy.GetViewer(Globals.env)
        print "set viewpoint, then press 'p'"
        Globals.viewer.Idle()

    return demofile

compare_bootstrap_correspondences = False# set to true and call with warp_root=False to compare warping derived trajectories to warping initial with bootstrapped correspondences

def get_warped_trajectory(seg_info, new_xyz, demofile, warp_root=True, plot=False, no_cmat=False):
    """
    @seg_info  : segment information from the h5 file for the segment with least tps fit cost.
    @new_xyz   : point cloud of the rope in the test situation.
    @warp_root : warp the root trajectory if True else warp the chosen segment's trajectory.
    
    @returns   : the warped trajectory for l/r grippers and the mini-segment information.
    """
    print "****WARP ROOT*** : ", warp_root
    print "****NO CMAT*** : ", no_cmat

    handles = []
    seg_xyz = seg_info["cloud_xyz"][:]    
    scaled_seg_xyz,  seg_params  = registration.unit_boxify(seg_xyz)
    scaled_new_xyz,  new_params  = registration.unit_boxify(new_xyz)

    if plot:
        handles.append(Globals.env.plot3(new_xyz, 5, (0, 0, 1)))
        handles.append(Globals.env.plot3(seg_xyz, 5, (1, 0, 0)))

    root_seg_name = seg_info['root_seg']
    root_segment = demofile[root_seg_name.value]
    root_xyz      = root_segment['cloud_xyz'][:]
    seg_root_cmat = seg_info['cmat'][:]
    if warp_root:
        scaled_root_xyz, root_params = registration.unit_boxify(root_xyz)

        if no_cmat:
            print "not using cmat for correspondences"
            f_root2new, _, corr_new2root = registration.tps_rpm_bij(scaled_root_xyz, scaled_new_xyz,
                                                                    plotting=5 if plot else 0, plot_cb=tpsrpm_plot_cb,
                                                                    rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=50,
                                                                    reg_init=10, reg_final=.01, old_xyz=root_xyz, new_xyz=new_xyz, 
                                                                    return_corr=True)
        else:

            ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            ## TODO : MAKE SURE THAT THE SCALING IS BEING DONE CORRECTLY HERE:
            ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!        
            f_root2new, _, corr_new2root = registration.tps_rpm_bootstrap(scaled_root_xyz, scaled_seg_xyz, scaled_new_xyz, seg_root_cmat, 
                                                                      plotting=5 if plot else 0, plot_cb=tpsrpm_plot_cb,
                                                                      rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=50,
                                                                      reg_init=10, reg_final=.01, old_xyz=root_xyz, new_xyz=new_xyz)
        f_warping = registration.unscale_tps(f_root2new, root_params, new_params)
        old_ee_traj  = root_segment['hmats']
        rgrip_joints = root_segment['r_gripper_joint'][:]
        lgrip_joints = root_segment['l_gripper_joint'][:]
        cmat         = corr_new2root

    else: ## warp to the chosen segment:
        f_seg2new, _, corr_new2seg = registration.tps_rpm_bij(scaled_seg_xyz, scaled_new_xyz, plot_cb=tpsrpm_plot_cb,
                                                              plotting=5 if plot else 0, rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=50,
                                                              reg_init=10, reg_final=.01, old_xyz=seg_xyz, new_xyz=new_xyz, 
                                                              return_corr=True)       
        f_warping = registration.unscale_tps(f_seg2new, seg_params, new_params)
        old_ee_traj = seg_info['hmats']
        rgrip_joints = root_segment['r_gripper_joint'][:]
        lgrip_joints = root_segment['l_gripper_joint'][:]
        
        cmat         = seg_root_cmat.dot(corr_new2seg)
        
        if compare_bootstrap_correspondences:
            scaled_root_xyz, root_params = registration.unit_boxify(root_xyz)
    
            f_root2new, _, corr_new2root = registration.tps_rpm_bootstrap(scaled_root_xyz, scaled_seg_xyz, scaled_new_xyz, seg_root_cmat, 
                                                                          plotting=5 if plot else 0, plot_cb=tpsrpm_plot_cb,
                                                                          rot_reg=np.r_[1e-4, 1e-4, 1e-1], n_iter=50,
                                                                          reg_init=10, reg_final=.01, old_xyz=root_xyz, new_xyz=new_xyz)
            root_warping = registration.unscale_tps(f_root2new, root_params, new_params)
            root_ee_traj  = root_segment['hmats']
            diff = 0
            for lr in 'lr':
                no_root_ee_traj        = f_warping.transform_hmats(old_ee_traj[lr][:])
                warped_root_ee_traj = root_warping.transform_hmats(root_ee_traj[lr][:])
                diff += np.linalg.norm(no_root_ee_traj - warped_root_ee_traj)
                handles.append(Globals.env.drawlinestrip(old_ee_traj[lr][:, :3, 3], 2, (1, 0, 0, 1)))
                handles.append(Globals.env.drawlinestrip(no_root_ee_traj[:, :3, 3], 2, (0, 1, 0, 1)))
                handles.append(Globals.env.drawlinestrip(warped_root_ee_traj[:, :3, 3], 2, (0, 1, 0, 1)))
            if plot:
                print 'traj norm differences:\t', diff
                Globals.viewer.Idle()
            

    if plot:
        handles.extend(plotting_openrave.draw_grid(Globals.env, f_warping.transform_points, new_xyz.min(axis=0) - np.r_[0, 0, .1],
                                                   new_xyz.max(axis=0) + np.r_[0, 0, .02], xres=.01, yres=.01, zres=.04))

    warped_ee_traj = {}
    #Transform the gripper trajectory here
    for lr in 'lr':
        new_ee_traj        = f_warping.transform_hmats(old_ee_traj[lr][:])
        warped_ee_traj[lr] = new_ee_traj

        if plot:
            handles.append(Globals.env.drawlinestrip(old_ee_traj[lr][:, :3, 3], 2, (1, 0, 0, 1)))
            handles.append(Globals.env.drawlinestrip(new_ee_traj[:, :3, 3], 2, (0, 1, 0, 1)))


    miniseg_starts, miniseg_ends = split_trajectory_by_gripper(rgrip_joints, lgrip_joints)
    return (cmat, warped_ee_traj, miniseg_starts, miniseg_ends, {'r':rgrip_joints, 'l':lgrip_joints})


def loop_body(task_params, demofile, choose_segment, knot, animate, curr_step=None, no_cmat=False):
    """
    Do the body of the main task execution loop (ie. do a segment).
    Arguments:
        curr_step is 0 indexed
        choose_segment is a function that returns the key in the demofile to the segment
        knot is the knot the rope is checked against
        new_xyz is the new pointcloud
        task_params is used for the only_original_segments argument

    seg_info is of the form:
    {'parent': The name of the segment from the action file that was chosen,
    'hmats': trajectories for the left and right grippers,
    'cloud_xyz': the new pointcloud,
    'cmat': the correspondence matrix or list of segments}
    #return {'found_knot': found_knot, 'seg_info': {'segment': segment, 'link2eetraj': link2eetraj, 'new_xyz': new_xyz}}
    return {'found_knot': found_knot, 'seg_info': seg_info}
    """
    redprint("Acquire point cloud")
    move_sim_arms_to_side()

    new_xyz = Globals.sim.observe_cloud(upsample=110)

    segment = choose_segment(demofile, new_xyz, 7)
    if segment is None:
        print "Got no segment while choosing a segment for warping."
        sys.exit(-1)

    seg_info      = demofile[segment]
    redprint("Getting warped trajectory...")
    cmat, warped_ee_traj, miniseg_starts, miniseg_ends, joint_traj = get_warped_trajectory(seg_info, new_xyz, demofile, 
                                                                                           warp_root=task_params.warp_root,
                                                                                           plot=task_params.animate,
                                                                                           no_cmat=no_cmat)
    success = True
    redprint("executing segment trajectory...")

    for (i_miniseg, (i_start, i_end)) in enumerate(zip(miniseg_starts, miniseg_ends)):
        lr_miniseg_traj = {}
        for lr in 'lr':
            ee_hmat_traj = warped_ee_traj[lr][:][i_start: i_end + 1]
            if arm_moved(ee_hmat_traj):
                lr_miniseg_traj[lr] = ee_hmat_traj

        yellowprint("\t Executing trajectory for segment %s, part %i using arms '%s'" % (segment, i_miniseg, lr_miniseg_traj.keys()))

        for lr in 'lr':
            gripper_open      = binarize_gripper(joint_traj[lr][i_start])
            prev_gripper_open = binarize_gripper(joint_traj[lr][i_start - 1]) if i_start != 0 else False
            if not set_gripper_sim(lr, gripper_open, prev_gripper_open, animate):
                redprint("Grab %s failed" % lr)
                success = False
        if not success:
            break

        if len(lr_miniseg_traj) > 0:
            success &= exec_traj_sim(lr_miniseg_traj, animate)

        if not success:
            break

    Globals.sim.settle(animate=animate)
    rope_nodes = Globals.sim.observe_cloud()
    found_knot = is_knot(rope_nodes)
    redprint("Segment %s result: %s" % (segment, success))
    seg_info_hash = {'parent': segment, 'hmats': warped_ee_traj, 'cloud_xyz': new_xyz, 'cmat': cmat}
    return {'found_knot': found_knot, 'seg_info':seg_info_hash}


def parse_arguments():
    import argparse

    usage = """
    Run {0} --help for a list of arguments
    Warning: This may write to the hdf5 demofile.
    See https://docs.google.com/document/d/17HmaCcXd5q9QST8P2rJMzuGCd3P0Rb1UdlNZATVWQz4/pub
    """.format(sys.argv[0])

    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument("h5file", type=str, help="The HDF5 file that contains the recorded demonstration segments.")

    parser.add_argument("--animation", action="store_true", help="Vizualize the robot and the simulation.")

    parser.add_argument("--select_manual", action="store_true",
                        help="Select the segments manually. If absent, then the segment will be automatically chosen.")

    parser.add_argument("--fake_data_transform", type=float, nargs=6, metavar=("tx", "ty", "tz", "rx", "ry", "rz"),
                        default=[0, 0, 0, 0, 0, 0], help="translation=(tx, ty, tz), axis-angle rotation=(rx, ry, rz)")

    parser.add_argument("--sim_init_perturb_radius", type=float, default=None,
                        help="How far to perturb the initial rope state. 0 in none, 0.1 is far.")

    parser.add_argument("--sim_init_perturb_num_points", type=int, default=7,
                        help="Perturb the rope state specified by fake_data_segment at this many points a distance of sim_init_perturb_radius.")

    parser.add_argument("--sim_desired_knot_name", type=str, default='K3a1',
                        help="Which knot the robot should tie. \"K3a1\" is an overhand knot.")

    parser.add_argument("--max_steps_before_failure", type=int, default=-1,
                        help="When not selecting manually (ie. automatic selection) it will declare failure after this many steps if the knot has not been detected.")

    parser.add_argument("--random_seed", type=int, default=None,
                        help="The random seed for the rope perturber. Using the same random seed (and keeping all of the other arguments the same too) allows initial perturbed rope states to be duplicated.")

    parser.add_argument("--log", type=str, default=None, help="Filename for the log file.")
    args = parser.parse_args()
    print "args =", args

    return args


def main():
    args = parse_arguments()
    if args.random_seed is not None:
        Globals.random_seed = args.random_seed
    choose_segment = find_closest_manual if args.select_manual else auto_choose
    params = TaskParameters(args.h5file, args.sim_desired_knot_name, animate=args.animation,
                            max_steps_before_failure=args.max_steps_before_failure, choose_segment=choose_segment,
                            log_name=args.log)
    result = do_single_task(params)
    print "Main results are", result


if __name__ == "__main__":
    main()

