## visualization in rviz. Uses John's python/brett2/ros_utils.py
from hd_utils import conversions as conv
from hd_utils import ros_utils as ru
import numpy as np

import roslib
import rospy
import geometry_msgs.msg as gm

rviz = ru.RvizWrapper.create()

# convert a list of transforms to pose array
def hmats_to_pose_array(hmats, frame_id):
    pose_array =  gm.PoseArray()
    for hmat in hmats:
        pose = conv.hmat_to_pose(hmat)
        pose_array.poses.append(pose)
    pose_array.header.frame_id = frame_id
    pose_array.header.stamp = rospy.Time.now()
    return pose_array


def draw_trajectory(frame_id, hmats, open_fracs=None, color=(0,0,1,0.5)):
    """
    Draws gripper trajectory in rviz. Init a ros-node before calling this.
    hold the return handles.
    """
    if open_fracs ==None:
        open_fracs = len(hmats)*[0]
    poses = hmats_to_pose_array(hmats, frame_id)
    handles = rviz.draw_trajectory(poses, open_fracs=open_fracs, color=color)
    return handles
