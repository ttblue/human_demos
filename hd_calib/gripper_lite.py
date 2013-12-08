import numpy as np, numpy.linalg as nlg
import networkx as nx
import itertools
import rospy

from hd_utils.colorize import *
from hd_utils.utils import avg_transform

import get_marker_transforms as gmt

class GripperLite ():
    """
    Since gripper has only only one marker, working with lighter version.
    Assumes gripper has only three markers and the calibration of ar to
    tool tip is known before hand.
    """
    
    # Change if need be
    parent_frame = 'camera1_rgb_optical_frame'
    
    cameras = None
    
    transform_graph = None
    lr = None

    ar_marker = None
    hydra_marker = None
    
    def __init__(self, lr, marker, x=0.07783, y=0.0, z=-0.04416, cameras=None):
        self.transform_graph = nx.Digraph()
        
        self.lr = lr
        self.ar_marker = marker
        self.cameras = cameras
        
        self.transform_graph.add_node(marker)
        self.transform_graph.add_node('tool_tip')
        self.transform_graph.add_edge(marker, 'tool_tip')
        
        tfm = np.eye(4)
        tfm[0:3,3] = np.array([x,y,z])
        
        self.transform_graph.edge[1]['tool_tip'] = tfm
        self.transform_graph.edge['tool_tip'][1] = nlg.inv(tfm)
        self.transform_graph.edge[1][1] = np.eye(4)
        self.transform_graph.edge['tool_tip']['tool_tip'] = np.eye(4)
        
    def get_ar_marker(self):
        return self.ar_marker
    
    def get_hydra_marker(self):
        return self.hydra_marker

    def set_cameras (self, cameras):
        self.cameras = cameras
        
    def get_rel_transform (m1, m2):
        return self.transform_graph.edge[m1][m2]
    
    def get_tooltip_transform (self, m, tfm):
        return tfm.dot(self.get_rel_transform(m, 'tool_tip'))
    
    def reset_gripper (self, lr, transforms, ar,hydra=None):
        """
        Resets gripper with new lr value and a list of transforms
        to populate transform graph.
        Each transform in list is dict with 'parent', 'child' and 'tfm'
        """
        self.lr = lr
        self.ar_marker = ar
        self.hydra_marker = hydra
        
        self.transform_graph = nx.DiGraph()
        for tfm in transforms:
            self.transform_graph.add_edge(tfm['parent'], tfm['child'])
            self.transform_graph.edge[tfm['parent']][tfm_parent['child']] = tfm['tfm']
            self.transform_graph.edge[tfm['child']][tfm_parent['parent']] = nlg.inv(tfm['tfm'])

        
        for i,j in itertools.combinations(sorted(self.transform_graph.nodes()), 2):
            if not self.transform_graph.has_edge(i,j):
                ij_path = nxa.shortest_path(self.transform_graph, i, j)
                Tij = self.transform_graph.edge[ij_path[0]][ij_path[1]]
                for p in xrange(2,len(ij_path)):
                    Tij = Tij.dot(self.transform_graph.edge[ij_path[p-1]][ij_path[p]])
                Tji = nlg.inv(Tij)
                
                self.transform_graph.add_edge(i,j)
                self.transform_graph.add_edge(j,i)
                self.transform_graph.edge[i][j] = Tij
                self.transform_graph.edge[j][i] = Tji
        
        for node in self.transform_graph.nodes_iter():
            self.transform_graph.edge[node][node] = np.eye(4)

    def get_saveable_transforms (self):
        transforms = []
        
        transforms.append({'parent':self.ar_marker, 'child':'tool_tip', 
                           'tfm':self.transform_graph.edge[self.ar_marker]['tool_tip']})
        
        if self.hydra_marker is not None:
            transforms.append({'parent':self.ar_marker, 'child':self.hydra_marker, 
                               'tfm':self.transform_graph.edge[self.ar_marker][self.hydra_marker]})
            
        return transforms

        
    def get_tfm_from_obs(self, marker, n_tfm = 10, n_avg=10):
        """
        Stores a bunch of AR readings.
        """
        all_tfms = []
        
        i = 0
        thresh = n_avg*2
        if rospy.get_name == '/unnamed':
            rospy.init_node('gripper_calib')
        sleeper = rospy.Rate(30)
        
        while i < n_tfm:
            raw_input(colorize("Getting transform %i out of %i. Hit return when ready."%(i,n_tfm), 'yellow', True))

            j = 0
            j_th = 0
            found = True
            htfms = []
            artfms = []
            while j < n_avg:
                blueprint("Averaging transform %i out of %i."%(j,n_avg))
                ar_tfms = self.cameras.get_ar_markers(markers=[self.ar_marker]);
                hyd_tfms = gmt.get_hydra_transforms(parent_frame=self.parent_frame, hydras=[marker]);
                
                if not ar_tfms and not hyd_tfms:
                    yellowprint("Could not find any transform.")
                    j_th += 1
                    if j_th < thresh:
                        continue
                    else:
                        found = False
                        break
                                
                artfms.append(ar_tfms[1])
                htfms.append(hyd_tfms[marker])

                j += 1
                sleeper.sleep()
            
            if found is False:
                yellowprint("Something went wrong; try again.")
                continue

            artfm = avg_transform(artfms)
            htfm = avg_transform(htfms)
            
            all_tfms.append(nlg.inv(artfm).dot(htfm))
            i += 1
            
        return avg_transform(all_tfms)
            

    def add_hydra(self, hydra_marker, tfm=None, ntfm=10,navg=20):
        """
        Assumes that this is to the AR marker on the gripper.
        """
        if tfm is not None:
            self.hydra_marker = hydra_marker
            self.transform_graph.add_edge(ar_marker, hydra_marker)
            self.transform_graph.edge[ar_marker][hydra_marker] = tfm
            self.transform_graph.edge[hydra_marker][ar_marker] = nlg.inv(inv)
            return
        
        tfm = self.get_tfm_from_obs(hydra_marker, n_tfm=ntfm, n_avg=navg)
        
        if hydra_marker != self.hydra_marker:
            self.transform_graph.add_edge(self.ar_marker, hydra_marker)
            self.transform_graph.add_edge(hydra_marker, self.ar_marker)
            self.transform_graph.add_edge('tool_tip', hydra_marker)
            self.transform_graph.add_edge(hydra_marker, 'tool_tip')
            self.hydra_marker = hydra_marker
            self.transform_graph.edge[hydra_marker][hydra_marker] = np.eye(4)

        self.transform_graph.edge[self.ar_marker][hydra_marker] = tfm
        self.transform_graph.edge[hydra_marker][self.ar_marker] = nlg.inv(tfm)
        
        ttar_tfm = self.transform_graph.edge['tool_tip'][self.ar_marker]
        self.transform_graph.edge['tool_tip'][hydra_marker] = ttar_tfm.dot(tfm)
        self.transform_graph.edge[hydra_marker]['tool_tip'] = nlg.inv(ttar_tfm.dot(tfm))


    def get_all_transforms(self, diff_cam=False):
        """
        From marker transforms given in parent_frame, get all transforms
        of all markers on gripper.
        """
        if self.cameras is None:
            redprint("Cameras not initialized. Could not get transforms.")
            return

        if diff_cam:
            ar_tfms = {}
            for i in range(self.cameras.num_cameras):
                tfms = self.cameras.get_ar_markers(camera=i, parent_frame=True, markers=[self.ar_marker])
                if tfms: ar_tfms[i] = tfms[self.ar_marker]
        else:
            ar_tfms = self.cameras.get_ar_markers(markers=[self.ar_marker])

        transforms = []
        
        if diff_cam:
            for i,tfm in ar_tfms.items():
                transforms.append({'parent':self.parent_frame,
                                   'child':'%sgripper_camera%i_tooltip'%(self.lr, i),
                                   'tfm':self.get_tooltip_transform(self.ar_marker, tfm)})
        elif ar_tfms:
            transforms.append({'parent':self.parent_frame,
                                'child':'%sgripper_ar_tooltip'%self.lr,
                                'tfm':self.get_tooltip_transform(self.ar_marker, ar_tfms[self.ar_marker])})
        if self.hydra_marker is not None:
            hyd_tfms = gmt.get_hydra_transforms(self.parent_frame, hydras=[self.hydra_marker])
            if hyd_tfms:
                tfm = hyd_tfms[self.hydra_marker]
                transforms.append({'parent':self.parent_frame,
                                   'child':'%sgripper_hydra_tooltip'%self.lr,
                                   'tfm':self.get_tooltip_transform(self.hydra_marker, tfm)})

        return transforms
