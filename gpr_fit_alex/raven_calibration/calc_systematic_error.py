import numpy as np, numpy.linalg as nlg
import pickle
import pylab as pl
from error_characterization import *

arm_side = 'R'

def get_robot_pose(timestamp, robot_poses):
    return min(range(len(robot_poses[arm_side])), key=lambda i: abs(robot_poses[arm_side][i][0] - timestamp))

data = pickle.load(open('raven_data.pkl'))

n_joints = len(data['robot_joints'][arm_side][0][1])
camera_to_robot_tf = data['camera_to_robot_tf']
robot_to_camera_tf = nlg.inv(camera_to_robot_tf)
camera_poses = []
robot_poses = []
robot_joints = np.empty((0,n_joints))

for ts_pose in data['camera_poses'][arm_side]:
    camera_poses.append(ts_pose[1])
    robot_pose_ind = get_robot_pose(ts_pose[0], data['robot_poses'])
    robot_poses.append(data['robot_poses'][arm_side][robot_pose_ind][1])
    robot_joints = np.r_[robot_joints, np.array([data['robot_joints'][arm_side][robot_pose_ind][1]])]

# split data into training and test data
camera_poses_test = [camera_poses[i] for i in range(len(camera_poses)) if i%4 == 0]
camera_poses = [camera_poses[i] for i in range(len(camera_poses)) if i%4 != 0]
robot_poses_test = [robot_poses[i] for i in range(len(robot_poses)) if i%4 == 0]
robot_poses = [robot_poses[i] for i in range(len(robot_poses)) if i%4 != 0]
robot_joints_test = np.take(robot_joints, [i for i in range(robot_joints.shape[0]) if i%4 == 0], axis=0)
robot_joints = np.take(robot_joints, [i for i in range(robot_joints.shape[0]) if i%4 != 0], axis=0)

n_data = len(camera_poses) # number of training data points (should be the same for robot_poses and robot_joints)

# original camera calibration transformed robot poses
orig_cal_robot_poses = [robot_to_camera_tf.dot(robot_pose) for robot_pose in robot_poses]
orig_cal_pose_error = calc_pose_error(camera_poses, orig_cal_robot_poses)
print "original camera calibration pose error for training data"
print "mean", np.mean(orig_cal_pose_error, axis=0)
print "std", np.std(orig_cal_pose_error, axis=0)


# Optimization problem to find better robot_to_camera_tf
new_robot_to_camera_tf = sys_correct_tf(camera_poses, robot_poses, robot_to_camera_tf)

# systematic (i.e. new camera calibration) corrected robot poses for training data
sys_robot_poses = [new_robot_to_camera_tf.dot(robot_pose) for robot_pose in robot_poses]
sys_pose_error = calc_pose_error(camera_poses, sys_robot_poses)
print "systematic corrected pose error for training data"
print "mean", np.mean(sys_pose_error, axis=0)
print "std", np.std(sys_pose_error, axis=0)

# Gaussian Process regression to estimate and apply the error as a function of the joint angles
sys_robot_poses_test = [new_robot_to_camera_tf.dot(robot_pose) for robot_pose in robot_poses_test]
# The following two lines is equivalent to the following
# gp_robot_poses_test = gp_correct_poses(camera_poses, sys_robot_poses, robot_joints, sys_robot_poses_test, robot_joints_test)
alphas = gp_correct_poses_precompute(camera_poses, sys_robot_poses, robot_joints)
gp_robot_poses_test = gp_correct_poses_fast(alphas, robot_joints, sys_robot_poses_test, robot_joints_test)


print "evaluating test data..."

# original camera calibration transformed robot poses
orig_cal_robot_poses_test = [robot_to_camera_tf.dot(robot_pose) for robot_pose in robot_poses_test]
orig_cal_pose_error = calc_pose_error(camera_poses_test, orig_cal_robot_poses_test)
print "original camera calibration pose error"
print "mean", np.mean(orig_cal_pose_error, axis=0)
print "std", np.std(orig_cal_pose_error, axis=0)

# systematic (i.e. new camera calibration) corrected robot poses
sys_pose_error = calc_pose_error(camera_poses_test, sys_robot_poses_test)
print "systematic corrected pose error"
print "mean", np.mean(sys_pose_error, axis=0)
print "std", np.std(sys_pose_error, axis=0)

# systematic and GP corrected robot poses
gp_pose_error = calc_pose_error(camera_poses_test, gp_robot_poses_test)
print "systematic and GP corrected pose error"
print "mean", np.mean(gp_pose_error, axis=0)
print "std", np.std(gp_pose_error, axis=0)

print "\nformatted error stats"
print_formatted_error_stat(orig_cal_pose_error)
print_formatted_error_stat(sys_pose_error)
print_formatted_error_stat(gp_pose_error)

pl.figure(1)
pl.plot(orig_cal_pose_error)
pl.figure(2)
pl.plot(sys_pose_error)
pl.figure(3)
pl.plot(gp_pose_error)

