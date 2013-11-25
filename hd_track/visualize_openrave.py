#! /usr/bin/env python
from __future__ import division
import cPickle
import openravepy as rave
import numpy as np
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--fname", help="joints file name.", required=True)
args = parser.parse_args()

traj = cPickle.load(open(args.fname, 'rb'))['mat']

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

print len(traj)
freq = 60.
for i in xrange(len(traj)):

    DOF = robot.GetActiveDOFValues()

    DOF[inds] = traj[i,:]
    robot.SetActiveDOFValues(DOF)
    time.sleep(1/freq)
