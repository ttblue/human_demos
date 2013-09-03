import rosbag
import numpy as np
import rospy
import pickle
import time
import os
import roslib
roslib.load_manifest('tf')
import tf
from sensor_msgs.msg import JointState
from hd_utils import conversions

def extract_joints(bag):
    traj = []
    stamps = []
    for (_, msg, _) in bag.read_messages(topics=['/joint_states']):
        traj.append(msg.position)
        stamps.append(msg.header.stamp.to_sec())
    assert len(traj) > 0
    names = msg.name
    return names, stamps, traj


old_time = 0
msgs = []
transforms = []
tf_listener = None

hydra = ''
gripper = ''
base_frame = 'base_footprint'
hydra_frame = 'hydra_base'

def log(msg):
    global old_time, msgs, tf_listener, hydra_frame, hydra, gripper, transforms
    now = time.time()
    hydra_sensor = 'hydra_%s'%(hydra)
    gripper_tool = '%s_gripper_tool_frame'%(gripper)
    if now - old_time > 1.0/30:
        msgs.append(msg)
        #print msg
        old_time = now
        trans, rot = tf_listener.lookupTransform(gripper_tool, hydra_sensor, rospy.Time(0))
        T_gt_hs = conversions.trans_rot_to_hmat(trans, rot)
        transforms.append(T_gt_hs)

def listen():
    rospy.Subscriber("joint_states", JointState, log)
    rospy.spin()

def process(name):
    global msgs, transforms

    joints = ['l_shoulder_pan_joint', 'l_shoulder_lift_joint', 'l_upper_arm_roll_joint', 'l_elbow_flex_joint', 'l_forearm_roll_joint', 'l_wrist_flex_joint', 'l_wrist_roll_joint', 'l_gripper_joint', 'r_shoulder_pan_joint', 'r_shoulder_lift_joint', 'r_upper_arm_roll_joint', 'r_elbow_flex_joint', 'r_forearm_roll_joint', 'r_wrist_flex_joint', 'r_wrist_roll_joint', 'r_gripper_joint']

    dic = {}
    sample_msg = msgs[0]
    indexes =[sample_msg.name.index(joint) for joint in joints]
    
    trajectories_list = [msg.position for msg in msgs]
    joint_states = []
    for i in xrange(len(trajectories_list)):
        trajectory = trajectories_list[i]
        filtered_traj = [trajectory[index] for index in indexes]
        joint_states.append(filtered_traj)
    joint_states_matrix = np.array(joint_states)
    d = {}
    for i, joint in enumerate(joints):
        d[joint] = i
    dic['trajectories'] = joint_states_matrix
    dic['mapping'] = d
    dic['transforms'] = transforms
    pickle.dump( dic, open( name, "wa" ) )
    
    #print d
    print ''
    print len(msgs)
    print len(transforms)
   
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("name")
parser.add_argument("gripper")
parser.add_argument("hydra")
args = parser.parse_args()

if __name__ == '__main__':
    rospy.init_node('log_data')
    global tf_listener, hydra, gripper
    tf_listener = tf.TransformListener()
    time.sleep(3)
    gripper = args.gripper
    hydra = args.hydra
    listen()
    process(args.name)
