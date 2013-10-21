# script to visualize recorded trajectories.
import roslib
import rospy
from hd_visualization import ros_vis as rv
import numpy as np
import cPickle as cp
import os.path as osp

def scale_tfms(Ts, s=1.):
    retT = []
    for t in Ts:
        st = t
        st[:,3] *= s
        retT.append(st)
    return retT

rospy.init_node('draw_demos')

## load data
demo_dir  = '/home/ankush/sandbox444/human_demos/hd_data/demos/obs_data'
demo_file = 'demo1.data'

fname = osp.join(demo_dir, demo_file)
dat   = cp.load(open(fname))

scale = 10.
ds    = 3

cam1_tfms  = scale_tfms([tt[0] for tt in dat['camera1']][::ds], scale)
cam2_tfms  = scale_tfms([tt[0] for tt in dat['camera2']][::ds], scale)
hydra_tfms = scale_tfms([tt[0] for tt in dat['hydra']][::50], scale)

frame = 'camera_link'
handles = []

handles.extend(rv.draw_trajectory(frame, cam1_tfms,  color=(1,0,0,0.5)))
handles.extend(rv.draw_trajectory(frame, cam2_tfms,  color=(0,1,0,0.5)))
handles.extend(rv.draw_trajectory(frame, hydra_tfms, color=(0,0,1,0.5)))

## draw curves connecting the grippers:
cam1_poses = rv.hmats_to_pose_array(cam1_tfms, frame)
cam1_colors = [(i/float(len(cam1_poses.poses)),0,0,0.5) for i in xrange(len(cam1_poses.poses))]
handles.append(rv.rviz.draw_curve(cam1_poses, rgba=cam1_colors))

cam2_poses = rv.hmats_to_pose_array(cam2_tfms, frame)
cam2_colors = [(0,i/float(len(cam1_poses.poses)),0,0.5) for i in xrange(len(cam2_poses.poses))]
handles.append(rv.rviz.draw_curve(cam2_poses, rgba=cam2_colors))

hydra_poses = rv.hmats_to_pose_array(hydra_tfms, frame)
hydra_colors = [(0,0,i/float(len(hydra_poses.poses)),0.5) for i in xrange(len(hydra_poses.poses))]
handles.append(rv.rviz.draw_curve(hydra_poses, rgba=hydra_colors))

rospy.spin()
