import trajoptpy, openravepy
import numpy as np
from hd_utils.colorize import *
import IPython as ipy

def animate_traj(traj, base_hmats, robot, pause=True, step_viewer=True, restore=True, callback=None):
    """make sure to set active DOFs beforehand"""
    if restore: _saver = openravepy.RobotStateSaver(robot)
    if step_viewer or pause: viewer = trajoptpy.GetViewer(robot.GetEnv())
    for (i,dofs) in enumerate(traj):
        if i%10 == 0: print "step %i/%i"%(i+1,len(traj))
        if callback is not None: callback(i)
        robot.SetActiveDOFValues(dofs)
        if base_hmats != None:
            robot.SetTransform(base_hmats[i])
        if pause: viewer.Idle()
        elif step_viewer and i%50 == 0: viewer.Step()
        
def animate_floating_traj(lhmats, rhmats, sim_env, pause=True, step_viewer=True, callback=None,step=5):
    assert len(lhmats)==len(rhmats), "I don't know how to animate trajectory with different lengths"
    if step_viewer or pause: viewer = trajoptpy.GetViewer(sim_env.sim.env)
    for i in xrange(len(lhmats)):
        if callback is not None: callback(i)
        sim_env.sim.grippers['r'].set_toolframe_transform(rhmats[i])
        sim_env.sim.grippers['l'].set_toolframe_transform(lhmats[i])
        if pause: viewer.Idle()
        elif step_viewer and not i%step: viewer.Step()

def animate_floating_traj_angs(lhmats, rhmats, ljoints, rjoints, sim_env, pause=True, step_viewer=True, callback=None,step=5):
    assert len(lhmats)==len(rhmats), "I don't know how to animate trajectory with different lengths"
    if step_viewer or pause: viewer = trajoptpy.GetViewer(sim_env.sim.env)
    eetrajs = {'l':np.array(lhmats), 'r':np.array(rhmats)}

    valid_inds = grippers_exceed_rope_length(sim_env, eetrajs, 0.15) #0.05
    min_gripper_dist = [np.inf] # minimum distance between gripper when the rope capsules are too far apart

    for i in xrange(len(lhmats)):
        if is_rope_pulled_too_tight(i, eetrajs, min_gripper_dist, valid_inds, sim_env):
            continue
        if callback is not None: callback(i)
        sim_env.sim.grippers['l'].set_toolframe_transform(lhmats[i])
        sim_env.sim.grippers['r'].set_toolframe_transform(rhmats[i])
        if ljoints!=None and rjoints!=None:
            sim_env.sim.grippers['l'].set_gripper_joint_value(ljoints[i])
            #print "l: ", ljoints[i]
            sim_env.sim.grippers['r'].set_gripper_joint_value(rjoints[i])
            #print "r: ", rjoints[i]
        if pause: viewer.Idle()
        elif step_viewer and not i%step: viewer.Step()
    if min_gripper_dist[0] != np.inf:
        yellowprint("Some steps of the trajectory were not executed because the gripper was pulling the rope too tight.")

def is_rope_pulled_too_tight(i_step, ee_trajs, min_gripper_dist, valid_inds, sim_env):
    if i_step == len(ee_trajs['l'])-1: #last step
        return is_rope_pulled_too_tight(i_step-1, ee_trajs, min_gripper_dist, valid_inds, sim_env)
    if valid_inds is None or valid_inds[i_step]: # gripper is not holding the rope or the grippers are not too far apart
        return False
    rope = sim_env.sim.rope
    trans = rope.GetTranslations()
    hhs = rope.GetHalfHeights()
    rots = rope.GetRotations()
    fwd_pts = (trans + hhs[:,None]*rots[:,:3,0])
    bkwd_pts = (trans - hhs[:,None]*rots[:,:3,0])
    pts_dists = np.apply_along_axis(np.linalg.norm, 1, fwd_pts[:-1] - bkwd_pts[1:])[:,None] # these should all be zero if the rope constraints are satisfied
    if np.any(pts_dists > 1.5*sim_env.sim.rope_params.radius): #2 optimal so far
        if i_step == 0:# or i_step == len(ee_trajs['l'])-1:
            return False
        min_gripper_dist[0] = min(min_gripper_dist[0], np.linalg.norm(ee_trajs['r'][i_step,:3,3] - ee_trajs['l'][i_step,:3,3]))
        grippers_moved_closer = np.linalg.norm(ee_trajs['r'][i_step+1,:3,3] - ee_trajs['l'][i_step+1,:3,3]) < min_gripper_dist[0]
        return (not grippers_moved_closer)
    return False

def grippers_exceed_rope_length(sim_env, ee_trajs, thresh):
    """
    Let min_length be the minimun length of the rope between the parts being held by the left and right gripper.
    This function returns a mask of the trajectory steps in which the distance between the grippers doesn't exceed min_length-thresh.
    If not both of the grippers are holding the rope, this function return None.
    """
    if sim_env.sim.constraints['l'] and sim_env.sim.constraints['r']:
        min_length = np.inf
        hs = sim_env.sim.rope.GetHalfHeights()
        for i_end in [0,-1]:
            for j_end in [0,-1]:
                # ipy.embed()
                i_cnt_l = sim_env.sim.constraints_inds['l'][i_end]
                i_cnt_r = sim_env.sim.constraints_inds['r'][j_end]
                if i_cnt_l > i_cnt_r:
                    i_cnt_l, i_cnt_r = i_cnt_r, i_cnt_l
                min_length = min(min_length, 2*hs[i_cnt_l+1:i_cnt_r].sum() + hs[i_cnt_l] + hs[i_cnt_r])
        valid_inds = np.apply_along_axis(np.linalg.norm, 1, (ee_trajs['r'][:,:3,3] - ee_trajs['l'][:,:3,3])) < min_length - thresh
        return valid_inds
    else:
        return None