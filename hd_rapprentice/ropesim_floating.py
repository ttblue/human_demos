import bulletsimpy
import numpy as np
from hd_utils import math_utils
from hd_rapprentice import retiming, resampling
from hd_utils.defaults import models_dir
import os.path as osp

def transform(hmat, p):
    return hmat[:3,:3].dot(p) + hmat[:3,3]


def retime_hmats(lhmats, rhmats, max_cart_vel=.02, upsample_time=.1):
    """
    retimes hmats (4x4 transforms) for left and right grippers
    """
    assert len(lhmats) == len(rhmats)
    cart_traj = np.empty((len(rhmats), 6))
    for i in xrange(len(lhmats)):
        cart_traj[i,:3] = lhmats[i][:3,3]
        cart_traj[i,3:] = rhmats[i][:3,3]
    times    = retiming.retime_with_vel_limits(cart_traj, np.repeat(max_cart_vel, 6))
    times_up = np.linspace(0, times[-1], times[-1]/upsample_time) if times[-1] > upsample_time else times
    lhmats_up = resampling.interp_hmats(times_up, times, lhmats)
    rhmats_up = resampling.interp_hmats(times_up, times, rhmats)
    return (lhmats_up, rhmats_up)


class FloatingGripper(object):
    def __init__(self, env, init_tf):
        gripper_fname = osp.join(models_dir, 'pr2_gripper.dae')
        self.env = env
        self.env.Load(gripper_fname)
        self.robot   = self.env.GetRobots()[-1]
        self.tt_link = self.robot.GetLinks()[-1]
        self.robot.SetTransform(init_tf)

        ## conversion transforms:
        self.tf_tt2base = np.linalg.inv(self.tt_link.GetTransform()).dot(init_tf)
        self.tf_ee2tt   = np.array([[  0,   0,  -1,   0],
                                    [  0,   1,   0,   0],
                                    [  1,   0,   0,   0],
                                    [  0,   0,   0,   1]])
        self.tf_tt2ee   = np.linalg.inv(self.tf_ee2tt)
        self.tf_ee2base = self.tf_ee2tt.dot(self.tf_tt2base)

    def set_toolframe_transform(self, tf_ee):
        tf_base = tf_ee.dot(self.tf_tt2base)
        self.robot.SetTransform(tf_base)

    def get_toolframe_transform(self):
        return self.tt_link.GetTransform()#.dot(self.tf_tt2ee)

    def get_endeffector_transform(self):
        return self.tt_link.GetTransform().dot(self.tf_tt2ee)

    def get_gripper_joint_value(self):
        return self.robot.GetDOFValues()[0]
    
    def set_gripper_joint_value(self, jval):
        self.robot.SetDOFValues([jval], [0])
        
    def in_grasp_region(self, pt):
        """
        checks if the point PT is in the graspable region of this gripper.
        """
        tol = .00
    
        l_finger = self.robot.GetLink("l_gripper_l_finger_tip_link")
        r_finger = self.robot.GetLink("l_gripper_r_finger_tip_link")

        def on_inner_side(pt, finger_lr):
            finger = l_finger
            closing_dir = np.array([0, -1, 0])
            
            local_inner_pt = np.array([0.234402, -0.299, 0])/20.
            if finger_lr == "r":
                finger = r_finger
                closing_dir *= -1
                local_inner_pt[1] *= -1
            inner_pt = transform(finger.GetTransform(), local_inner_pt)
            return self.get_endeffector_transform()[:3,:3].dot(closing_dir).dot(pt - inner_pt) > 0
    
        # check that pt is behind the gripper tip
        pt_local = transform(np.linalg.inv(self.get_endeffector_transform()), pt)
        if pt_local[2] > .03 + tol:
            return False
    
        # check that pt is within the finger width
        if abs(pt_local[0]) > .01 + tol:
            return False
    
        # check that pt is between the fingers
        if not on_inner_side(pt, "l") or not on_inner_side(pt, "r"):
            return False
    
        return True


class FloatingGripperSimulation(object):
    def __init__(self, env):
        self.env      = env
        self.grippers = None
        self.__init_grippers__()
        self.bt_env   = None
        self.bt_robot = None
        self.rope     = None
        self.constraints = {"l": [], "r": []}

        self.rope_params = bulletsimpy.CapsuleRopeParams()
        #radius: A larger radius means a thicker rope.
        self.rope_params.radius = 0.005
        #angStiffness: a rope with a higher angular stifness seems to have more resistance to bending.
        #orig self.rope_params.angStiffness = .1
        self.rope_params.angStiffness = .1
        #A higher angular damping causes the ropes joints to change angle slower.
        #This can cause the rope to be dragged at an angle by the arm in the air, instead of falling straight.
        #orig self.rope_params.angDamping = 1
        self.rope_params.angDamping = 1
        #orig self.rope_params.linDamping = .75
        #Not sure what linear damping is, but it seems to limit the linear accelertion of centers of masses.
        self.rope_params.linDamping = .75
        #Angular limit seems to be the minimum angle at which the rope joints can bend.
        #A higher angular limit increases the minimum radius of curvature of the rope.
        self.rope_params.angLimit = .4
        #TODO--Find out what the linStopErp is
        #This could be the tolerance for error when the joint is at or near the joint limit
        self.rope_params.linStopErp = .2

    def __init_grippers__(self):
        """
        load the gripper models
        """
        rtf   = np.eye(4)
        ltf   = np.eye(4)
        rtf[0:3,3] = [0.5, 0,1]
        ltf[0:3,3] = [-0.5,0,1]

        self.grippers = {'l' : FloatingGripper(self.env, ltf),
                         'r' : FloatingGripper(self.env, rtf)}

    def create(self, rope_pts):
        bulletsimpy.sim_params.friction = 1
        self.bt_env   = bulletsimpy.BulletEnvironment(self.env, [])
        self.bt_env.SetGravity([0, 0, -9.8])
        self.bt_grippers = {lr : self.bt_env.GetObjectByName(self.grippers[lr].robot.GetName()) for lr in 'lr'}
        self.rope        = bulletsimpy.CapsuleRope(self.bt_env, 'rope', rope_pts, self.rope_params)

        # self.rope.UpdateRave()
        # self.env.UpdatePublishedBodies()
        # trajoptpy.GetViewer(self.env).Idle()
        self.settle()

    def step(self):
        for lr in 'lr':
            self.bt_grippers[lr].UpdateBullet()
        self.bt_env.Step(.01, 200, .005)
        self.rope.UpdateRave()
        self.env.UpdatePublishedBodies()

    def settle(self, max_steps=100, tol=.001, animate=False):
        """Keep stepping until the rope doesn't move, up to some tolerance"""
        prev_nodes = self.rope.GetNodes()
        for i in range(max_steps):
            self.bt_env.Step(.01, 200, .005)
            if animate:
                self.rope.UpdateRave()
                self.env.UpdatePublishedBodies()
            if i % 10 == 0 and i != 0:
                curr_nodes = self.rope.GetNodes()
                diff = np.sqrt(((curr_nodes - prev_nodes)**2).sum(axis=1))
                if diff.max() < tol:
                    break
                prev_nodes = curr_nodes
        self.rope.UpdateRave()
        self.env.UpdatePublishedBodies()
        print "settled in %d iterations" % (i+1)

    def observe_cloud(self, upsample=0):
        pts = self.rope.GetControlPoints()
        if upsample == 0:
            return pts
        lengths = np.r_[0, self.rope.GetHalfHeights() * 2]
        summed_lengths = np.cumsum(lengths)
        assert len(lengths) == len(pts)
        return math_utils.interp2d(np.linspace(0, summed_lengths[-1], upsample*len(pts)), summed_lengths, pts)

    def grab_rope(self, lr):
        """Grab the rope with the gripper in simulation and return True if it grabbed it, else False."""
        #GetNodes returns some sort of list of the positions of the centers of masses of the capsules.
        #GetControlPoints seems to return some sort of list of the verticies (bend points) of the rope).
        nodes, ctl_pts = self.rope.GetNodes(), self.rope.GetControlPoints()

        graspable_nodes = np.array([self.grippers[lr].in_grasp_region(n) for n in nodes])
        graspable_ctl_pts = np.array([self.grippers[lr].in_grasp_region(n) for n in ctl_pts])
        graspable_inds = np.flatnonzero(np.logical_or(graspable_nodes, np.logical_or(graspable_ctl_pts[:-1], graspable_ctl_pts[1:])))
        #print 'graspable inds for %s: %s' % (lr, str(graspable_inds))
        if len(graspable_inds) == 0:
            #No part close enough to the gripper to grab, so return False.
            return False

        robot_link = self.grippers[lr].robot.GetLink("l_gripper_l_finger_tip_link")
        rope_links = self.rope.GetKinBody().GetLinks()
        for i_node in graspable_inds:
            for i_cnt in range(max(0, i_node-1), min(len(nodes), i_node+2)):
                cnt = self.bt_env.AddConstraint({
                    "type": "generic6dof",
                    "params": {
                        "link_a": robot_link,
                        "link_b": rope_links[i_cnt],
                        "frame_in_a": np.linalg.inv(robot_link.GetTransform()).dot(rope_links[i_cnt].GetTransform()),
                        "frame_in_b": np.eye(4),
                        "use_linear_reference_frame_a": False,
                        "stop_erp": 0.8,
                        "stop_cfm": 0.1,
                        "disable_collision_between_linked_bodies": True,
                    }
                })
                self.constraints[lr].append(cnt)
        return True

    def release_rope(self, lr):
        print 'RELEASE: %s (%d constraints)' % (lr, len(self.constraints[lr]))
        for c in self.constraints[lr]:
            self.bt_env.RemoveConstraint(c)
        self.constraints[lr] = []
