#!/usr/bin/env python
"""
Script to visualize the rope and robot state in openrave viewer.
"""
import os.path as osp
import numpy as np
from numpy.linalg import norm
import openravepy, time

from hd_utils.utils import make_perp_basis
from hd_utils.defaults import cad_files_dir



class RopePR2Viz(object):

    def __init__(self):
        self.env  = openravepy.Environment()
        self.env.StopSimulation()
        self.env.Load(osp.join(cad_files_dir, 'table_sim.xml'))
        self.env.Load("robots/pr2-beta-static.zae")
        self.env.SetViewer('qtcoin', False)
        print self.env.GetViewer()

        self.robot = self.env.GetRobots()[0]
        
        self.cam_tfm = openravepy.matrixFromAxisAngle(np.pi*np.array([-0.7,0.7,0]))
        self.cam_tfm = openravepy.matrixFromAxisAngle([0,0,np.pi]).dot(self.cam_tfm)
        self.cam_tfm[:3,3] = np.array([0.5,0.0, 2])
        
        self.rope  = None
        #self.hide_links()
        

    def hide_links(self):
        for link in self.robot.GetLinks():
            if 'gripper' not in link.GetName():
                link.SetVisible(False)


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
            time.sleep(0.01) ## << required for correct rendering
    
        self.add_rope_to_rave(control_pts)
        
    def update_rope_links(self, control_pts):
        tfms, _ = self.create_rope_tfms(control_pts)
        for i in xrange(len(tfms)):    
            rope_link = self.env.GetKinBody('rope').GetLinks()[i]
            rope_link.SetTransform(tfms[i])
        self.rope = self.env.GetKinBody('rope')

            
    def set_robot_pose(self, dofs, robot_tfm):
        """
        sets the all the dofs of the robot and its base transform.
        """ 
        assert len(dofs) == self.robot.GetDOF()
        self.robot.SetDOFValues(dofs, range(len(dofs)))
        self.robot.SetTransform(robot_tfm)
        
    def get_env_snapshot(self, resX=640, resY=480, cam_tfm=None):
        if cam_tfm==None:
            cam_tfm = self.cam_tfm
        return self.env.GetViewer().GetCameraImage(resX, resY, cam_tfm, [640, 640, resX//2, resY//2])
        
        