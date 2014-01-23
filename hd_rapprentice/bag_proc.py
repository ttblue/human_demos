import fastrapp
import numpy as np
import cv2
import openravepy
import os.path as osp

from hd_utils import func_utils
from hd_utils.defaults import tfm_head_dof
import ros2rave

def extract_joints(bag):
    """returns (names, traj) 
    """
    traj = []
    stamps = []
    for (_, msg, _) in bag.read_messages(topics=['/joint_states']):        
        traj.append(msg.position)
        stamps.append(msg.header.stamp.to_sec())
    assert len(traj) > 0
    names = msg.name
    return names, stamps, traj
    
def extract_joy(bag):
    """sounds morbid    
    """

    stamps = []
    meanings = []
    button2meaning = {
        12: "look",
        0: "start",
        3: "stop",
        7: "l_open",
        5: "l_close",
        15: "r_open",
        13: "r_close",
        14: "done"
    }
    check_buttons = button2meaning.keys()
    message_stream = bag.read_messages(topics=['/joy'])
    (_,last_msg,_) = message_stream.next()
    for (_, msg, _) in message_stream:
        for i in check_buttons:
            if msg.buttons[i] and not last_msg.buttons[i]:
                stamps.append(msg.header.stamp.to_sec())
                meanings.append(button2meaning[i])
        last_msg = msg
        
    return stamps, meanings

        
def find_disjoint_subsequences(li, seq):
    """
    Returns a list of tuples (i,j,k,...) so that seq == (li[i], li[j], li[k],...)
    Greedily find first tuple, then second, etc.
    """
    subseqs = []
    cur_subseq_inds = []
    for (i_el, el) in enumerate(li):
        if el == seq[len(cur_subseq_inds)]:
            cur_subseq_inds.append(i_el)
            if len(cur_subseq_inds) == len(seq):
                subseqs.append(cur_subseq_inds)
                cur_subseq_inds = []
    return subseqs
    
def joy_to_annotations(stamps, meanings):
    """return a list of dicts giving info for each segment
    [{"look": 1234, "start": 2345, "stop": 3456},...]
    """
    out = []
    ind_tuples = find_disjoint_subsequences(meanings, ["look","start","stop"])
    for tup in ind_tuples:
        out.append({"look":stamps[tup[0]], "start":stamps[tup[1]], "stop":stamps[tup[2]]})
    
    done_inds = [i for (i,meaning) in enumerate(meanings) if meaning=="done"]
    for ind in done_inds:
        out.append({"done":None,"look":stamps[ind], "start":stamps[ind], "stop":stamps[ind]+1})
    
    return out

def add_kinematics_to_group(group, linknames, manipnames, jointnames, robot):
    "do forward kinematics on those links"
    if robot is None: robot = get_robot()
    r2r = ros2rave.RosToRave(robot, group["joint_states"]["name"])
    link2hmats = dict([(linkname, []) for linkname in linknames])
    links = [robot.GetLink(linkname) for linkname in linknames]
    rave_traj = []
    rave_inds = r2r.rave_inds
    for ros_vals in group["joint_states"]["position"]:
        r2r.set_values(robot, ros_vals)
        rave_vals = r2r.convert(ros_vals)
        robot.SetDOFValues(rave_vals, rave_inds)
        rave_traj.append(rave_vals)
        for (linkname,link) in zip(linknames, links):
            link2hmats[linkname].append(link.GetTransform())
    for (linkname, hmats) in link2hmats.items():
        group.create_group(linkname)
        group[linkname]["hmat"] = np.array(hmats)      
        
    rave_traj = np.array(rave_traj)
    rave_ind_list = list(rave_inds)
    for manipname in manipnames:
        arm_inds = robot.GetManipulator(manipname).GetArmIndices()
        group[manipname] = rave_traj[:,[rave_ind_list.index(i) for i in arm_inds]]
        
    for jointname in jointnames:
        joint_ind = robot.GetJointIndex(jointname)
        group[jointname] = rave_traj[:,rave_ind_list.index(joint_ind)]
        
    
    
    
@func_utils.once
def get_robot():
    env = openravepy.Environment()
    env.Load("robots/pr2-beta-static.zae")
    robot = env.GetRobots()[0]
    return robot
    
