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
import openravepy

from hd_utils import yes_or_no
from hd_utils.utils import make_perp_basis
from hd_utils.colorize import *
from hd_utils.defaults import demo_files_dir, hd_data_dir, cad_files_dir



class RopePR2Viz(object):

    def __init__(self):
        self.env  = openravepy.Environment()
        self.env.StopSimulation()
        self.env.Load(osp.join(cad_files_dir, 'table_sim.xml'))
        self.env.Load("robots/pr2-beta-static.zae")
        self.robot = self.env.GetRobots()[0]
        self.rope  = None


    def set_robot_pose(self, dofs, robot_tfm):
        """
        sets the all the dofs of the robot and its base transform.
        """ 
        assert len(dofs) == self.robot.GetDOF()
        self.robot.SetDOFValues(dofs, range(len(dofs)))
        self.robot.SetTransform(robot_tfm)


    def create_rope_xml(self, name, radius, lengths, rgba=(0.89,0.46,0,1)):
        xml  =  "<KinBody name=\"%s\">"%name
        for i in xrange(len(lengths)):
            xml += "<Body name=\"%s_%d\" type=\"static\"><Geom type=\"cylinder\">"%(name, i)
            xml += "<radius>%f</radius>"%radius
            xml += "<height>%f</height>" % lengths[i]
            xml += "<RotationAxis>0 0 1 90</RotationAxis>"
            xml += "<diffuseColor>%f %f %f %f</diffuseColor>"%rgba
            xml += "</Geom></Body>"
        xml += "</KinBody>"
        return xml


    def create_rope_tfms(self, control_pts):
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

    def add_rope_to_rave(self, control_pts):
        tfms, lengths = self.create_rope_tfms(control_pts)
        rope_xml      = self.create_rope_xml('rope', radius=0.01, lengths=lengths)
        self.env.LoadData(rope_xml)
        
        for i in xrange(len(tfms)):    
            rope_link = self.env.GetKinBody('rope').GetLinks()[i]
            rope_link.SetTransform(tfms[i])
        self.rope = self.env.GetKinBody('rope')
    

    def update_rope(self, control_pts):
        if self.rope != None:
            self.env.Remove(self.env.GetKinBody('rope'))
        self.add_rope_to_rave(control_pts)
            
