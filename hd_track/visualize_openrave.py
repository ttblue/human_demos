#! /usr/bin/env python
import pickle
import openravepy as rave
import numpy as np
import time



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("matrix_name")
parser.add_argument("dictionary_name")
args = parser.parse_args()

traj = pickle.load(open(args.matrix_name, 'r'))
dic = pickle.load(open(args.dictionary_name, 'r'))
e = rave.Environment()
e.Load("robots/pr2-beta-static.zae")

robot = e.GetRobots()[0]
e.SetViewer('qtcoin')

lm = robot.GetManipulator('leftarm')
rm = robot.GetManipulator('rightarm')

linds = lm.GetArmIndices()
rinds = rm.GetArmIndices()

inds = np.append(linds, 22)
inds = np.append(inds, rinds)
inds = np.append(inds, 34)

joints = robot.GetJoints(inds)
joint_names = [joint.GetName() for joint in joints]
print inds

#column_indices = [dic[name] for name in joint_names]
print len(traj)
for i in xrange(len(traj)):

    DOF = robot.GetActiveDOFValues()

    DOF[inds] = traj[i,:]
    robot.SetActiveDOFValues(DOF)
    time.sleep(0.033333)
