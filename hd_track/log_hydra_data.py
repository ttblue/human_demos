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


T_rtool_rhydra = np.array([[ 0.72076,  0.68234,  0.12211,  0.4814 ],
 [-0.68672,  0.72687, -0.00826,  0.01968],
 [-0.09439, -0.0779,   0.99248,  0.83744],
 [ 0.,       0.,       0.,       1.,     ]])

old_time = 0
msgs = []
tool_transforms = []
hydra_transforms = []
tf_listener = None

hydra = ''
gripper = ''
base_frame = 'base_footprint'
hydra_frame = 'hydra_base'

def log(msg):
    global old_time, msgs, tf_listener, base_frame, hydra_frame, hydra, gripper, tool_transforms, hydra_transforms
    now = time.time()
    hydra_sensor = 'hydra_%s'%(hydra)
    gripper_tool = '%s_gripper_tool_frame'%(gripper)
    if now - old_time > 1.0/30:
        msgs.append(msg)
        print "logging"
        old_time = now
        trans_g, rot_g = tf_listener.lookupTransform(base_frame, gripper_tool, rospy.Time(0))
        trans_h, rot_h = tf_listener.lookupTransform(base_frame, hydra_sensor, rospy.Time(0))
        T_g = conversions.trans_rot_to_hmat(trans_g, rot_g)
        T_h = conversions.trans_rot_to_hmat(trans_h, rot_h)
        tool_transforms.append(T_g)
        hydra_transforms.append(T_h)
        

def listen():
    rospy.Subscriber("joint_states", JointState, log)
    rospy.spin()

def process(name):
    global msgs, hydra_transforms, tool_transforms

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
    #dic['trajectories'] = joint_states_matrix
    #dic['mapping'] = d
    dic['T_bh'] = hydra_transforms
    dic['T_bg'] = tool_transforms
    pickle.dump( dic, open( name, "wa" ) )
    
    #print d
    print ''
    print len(hydra_transforms)
    print len(tool_transforms)
   
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', help="Name of the log file", required=True, type=str)
parser.add_argument('--gripper', help="PR2 gripper to track {l, r}", required=True, type=str)
parser.add_argument('--hydra', help="Hydra sensor to track {left, right}", required=True, type=str)
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
