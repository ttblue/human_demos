#!/usr/bin/env python
"""
Script to visualize the rope and robot state in openrave viewer.
"""
import argparse
usage=""
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("--demo_type", type=str)
args = parser.parse_args()

import os, numpy as np, h5py, time, os.path as osp
import cPickle
import numpy as np
from numpy.linalg import norm

import cloudprocpy, trajoptpy, openravepy

from hd_rapprentice import registration, animate_traj, ros2rave, \
     plotting_openrave, task_execution, \
     ropesim

from hd_utils import yes_or_no
from hd_utils.utils import make_perp_basis
from hd_utils.colorize import *
from hd_utils.defaults import demo_files_dir, hd_data_dir, cad_files_dir


class Globals:
    env   = None
    robot = None
    rope  = None

def get_random_rope_nodes(n=10):
    """
    return nx3 matrix of rope nodes:
    """
    nodes = np.c_[np.linspace(0,1,n), np.linspace(0,1,n), np.linspace(0,1,n)]
    nodes += np.random.randn(n,3)/20.
    return nodes


def set_robot_pose(dofs, robot_tfm):
    """
    sets the all the dofs of the robot and its base transform.
    """ 
    assert len(dofs) == Globals.robot.GetDOF()
    Globals.robot.SetDOFValues(dofs, range(len(dofs)))
    Globals.robot.SetTransform(robot_tfm)


def create_rope_xml(name, radius, lengths):
  xml  =  "<KinBody name=\"%s\">"%name
  for i in xrange(len(lengths)):
    xml += "<Body name=\"%s_%d\" type=\"static\"><Geom type=\"cylinder\">"%(name, i)
    xml += "<radius>%f</radius>"%radius
    xml += "<height>%f</height>" % lengths[i]
    xml += "<RotationAxis>0 0 1 90</RotationAxis>"
    xml += "</Geom></Body>"
  xml += "</KinBody>"
  return xml


def create_rope_tfms(control_pts):
    """
    control_pts : nx3 matrix
    """
    nLinks = len(control_pts)-1
    transforms = []
    lengths    = []
    for i in xrange(nLinks):
        pt0, pt1 = control_pts[i], control_pts[i+1] 
        midpt = (pt0+pt1)/2
        diff  = pt1-pt0

        rotation = make_perp_basis(diff)
        transform = np.eye(4)
        transform[:3,:3] = rotation
        transform[:3,3]  = midpt
        length   = norm(diff)

        transforms.append(transform)
        lengths.append(length)

    return (transforms, lengths)


def add_rope_to_rave(control_pts):
    tfms, lengths = create_rope_tfms(control_pts)
    rope_xml      = create_rope_xml('rope', radius=0.01, lengths=lengths)
    Globals.env.LoadData(rope_xml)

    for i in xrange(len(tfms)):
        rope_link = Globals.env.GetKinBody('rope').GetLinks()[i]
        rope_link.SetTransform(tfms[i])
    Globals.rope = Globals.env.GetKinBody('rope')


def update_rope(control_pts):
    if Globals.rope != None:
        Globals.env.Remove(Globals.env.GetKinBody('rope'))
    add_rope_to_rave(control_pts)


def main():
    Globals.env  = openravepy.Environment()
    Globals.env.StopSimulation()
    Globals.env.Load("robots/pr2-beta-static.zae")
    Globals.env.SetViewer('qtcoin')
    Globals.robot = Globals.env.GetRobots()[0]
    
    Globals.env.Load(osp.join(cad_files_dir, 'table_sim.xml'))
    body = Globals.env.GetKinBody('table')
    #Globals.viewer.SetTransparency(body,0.4)

#         if "hitch_pos" in demofile[args.fake_data_demo][args.fake_data_segment].keys():
#             Globals.env.Load(osp.join(cad_files_dir, 'hitch.xml'))
#             hitch_pos = demofile[args.fake_data_demo][args.fake_data_segment]['hitch_pos']
#             hitch_body = Globals.env.GetKinBody('hitch')
#             table_body = Globals.env.GetKinBody('table')
#             if init_tfm != None:
#                 hitch_pos = init_tfm[:3,:3].dot(hitch_pos) + init_tfm[:3,3]
#             hitch_tfm = hitch_body.GetTransform()
#             hitch_tfm[:3, 3] = hitch_pos
#             hitch_height = hitch_body.GetLinks()[0].GetGeometries()[0].GetCylinderHeight()
#             table_z_extent = table_body.GetLinks()[0].GetGeometries()[0].GetBoxExtents()[2] 
#             table_height = table_body.GetLinks()[0].GetGeometries()[0].GetTransform()[2, 3]
#             hitch_tfm[2, 3] = table_height + table_z_extent + hitch_height/2.0
#             hitch_body.SetTransform(hitch_tfm)
#         hitch = Globals.env.GetKinBody('hitch')
    while True:
        update_rope(get_random_rope_nodes(10))
        time.sleep(1)

if __name__ == "__main__":
    main()
