import numpy as np, numpy.linalg as nlg
import networkx as nx
import itertools
import rospy

from hd_utils.colorize import *
from hd_utils.utils import avg_transform
import networkx as nx, networkx.algorithms as nxa

import get_marker_transforms as gmt

class GripperLite ():
    """
    Since gripper has only one marker, working with lighter version.
    Assumes gripper has only one markers and the calibration of ar to
    tool tip is known before hand.
    """
    
    # Change if need be
    parent_frame = 'camera1_rgb_optical_frame'
    
    cameras = None
    
    transform_graph = None
    lr = None

    ar_marker = None
    hydra_marker = None
    
    def __init__(self, lr, marker, trans_marker_tooltip, cameras=None):
        '''
        Input: lr -- left or right; marker -- marker id, trans_marker_tooltip
        '''
        self.transform_graph = nx.DiGraph()
        
        self.lr = lr
        self.ar_marker = marker
        self.cameras = cameras
        
        self.transform_graph.add_node(marker)
        self.transform_graph.add_node('tool_tip')
        self.transform_graph.add_edge(marker, 'tool_tip')
        
        tfm = np.eye(4)
        tfm[0:3,3] = np.array(trans_marker_tooltip)
        
        self.transform_graph.edge[marker]['tool_tip'] = tfm
        self.transform_graph.edge['tool_tip'][marker] = nlg.inv(tfm)
        self.transform_graph.edge[marker][marker] = np.eye(4)
        self.transform_graph.edge['tool_tip']['tool_tip'] = np.eye(4)
        
    def get_ar_marker(self):
        return self.ar_marker
    
    def get_hydra_marker(self):
        return self.hydra_marker

    def set_cameras (self, cameras):
        self.cameras = cameras
        
    def get_rel_transform (self, m1, m2):
        return self.transform_graph.edge[m1][m2]
    
    def get_tooltip_transform (self, m, tfm):
        '''
        tfm is T_reference_m
        So this returns T_reference_tooltip
        '''
        return tfm.dot(self.get_rel_transform(m, 'tool_tip'))
    
    def reset_gripper (self, lr, transforms, ar, hydra=None):
        """
        Resets gripper with new lr value and a list of transforms
        to populate transform graph. Input also contains ar marker and hydra
        Each transform in list is dict with 'parent', 'child' and 'tfm'
        The resulting graph edges will contain transform between every pair of nodes in the same connected subgraph
        """
        self.lr = lr
        self.ar_marker = ar
        self.hydra_marker = hydra
        
        self.transform_graph = nx.DiGraph()
        for tfm in transforms:
            self.transform_graph.add_edge(tfm['parent'], tfm['child'])
            self.transform_graph.edge[tfm['parent']][tfm['child']] = tfm['tfm']
            self.transform_graph.edge[tfm['child']][tfm['parent']] = nlg.inv(tfm['tfm'])

        
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
        '''
        Save all the transforms from gripper's ar_marker to tooltip, and from ar_marker to hydra_marker
        '''
        transforms = []
        
        transforms.append({'parent':self.ar_marker, 'child':'tool_tip', 
                           'tfm':self.transform_graph.edge[self.ar_marker]['tool_tip']})
        
        if self.hydra_marker is not None:
            transforms.append({'parent':self.ar_marker, 'child':self.hydra_marker, 
                               'tfm':self.transform_graph.edge[self.ar_marker][self.hydra_marker]})
            
        return transforms

        
    def get_tfm_from_obs(self, hydra_marker, n_tfm=15, n_avg=30):
        """
        Stores a bunch of readings from sensors.
        """
        all_tfms = []
        
        if rospy.get_name == '/unnamed':
            rospy.init_node('gripper_calib')
        sleeper = rospy.Rate(30)
    
        i = 0
        n_attempts_max = n_avg*2
        while i < n_tfm:
            raw_input(colorize("Getting transform %i out of %i. Hit return when ready."%(i, n_tfm-1), 'yellow', True))

            j = 0
            n_attempts_effective = 0;
            hyd_tfms = []
            ar_tfms = []
            while j < n_attempts_max:
                j += 1
                ar_tfms_j = self.cameras.get_ar_markers(markers=[self.ar_marker]);
                hyd_tfms_j = gmt.get_hydra_transforms(parent_frame=self.parent_frame, hydras=[hydra_marker]);
                
                if not ar_tfms_j or not hyd_tfms_j:
                    if not ar_tfms_j:
                        yellowprint("Could not find required ar markers from " + str(self.ar_marker))
                    else:
                        yellowprint("Could not find required hydra transforms from " + str(hydra_marker))
                    continue
                                
                ar_tfms.append(ar_tfms_j[self.ar_marker])
                hyd_tfms.append(hyd_tfms_j[hydra_marker])
                
                blueprint('\tGetting averaging transform : %d of %d ...'%(n_attempts_effective, n_avg-1))
                n_attempts_effective += 1 
                
                if n_attempts_effective == n_avg:
                    break
                
                sleeper.sleep()
            
            if n_attempts_effective < n_avg:
                yellowprint("Not enough transforms were collected; try again")
                continue

            ar_tfm = avg_transform(ar_tfms)
            hyd_tfm = avg_transform(hyd_tfms)

            all_tfms.append(nlg.inv(ar_tfm).dot(hyd_tfm))
            i += 1
            
        return avg_transform(all_tfms)
            

    def add_hydra(self, hydra_marker, tfm=None, ntfm=15,navg=30):
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
        diff_cam: whether multiple cameras are used. 

        """
        if self.cameras is None:
            redprint("Cameras not initialized. Could not get transforms.")
            return

        if diff_cam:
            ar_tfms = {}
            for i in range(self.cameras.num_cameras):
                # parent_frame=True, so tfms is the transform from parent_frame (camera1) to ar_marker
                tfms = self.cameras.get_ar_markers(camera=i, parent_frame=True, markers=[self.ar_marker])
                if tfms: ar_tfms[i] = tfms[self.ar_marker]
        else:
            # default camera=0, so even parent_frame=False, ar_tfms still store the transform from camera1 to ar_marker
            ar_tfms = self.cameras.get_ar_markers(markers=[self.ar_marker]) 

        transforms = []
        
        # return T_camera_tooltip (if diff_cam, to different camera; otherwise to camera0)
        if diff_cam:
            for i,tfm in ar_tfms.items():
                # each tfm is T_camera1_gripper_ar_marker
                # get_tooltip_transform(self.ar_marker, tfm) returns T_camerai_tooltip
                transforms.append({'parent':self.parent_frame,
                                   'child':'%sgripper_camera%i_tooltip'%(self.lr, i+1),
                                   'tfm':self.get_tooltip_transform(self.ar_marker, tfm)
                                  })
        elif ar_tfms:
            transforms.append({'parent':self.parent_frame,
                               'child':'%sgripper_ar_tooltip'%self.lr,
                               'tfm':self.get_tooltip_transform(self.ar_marker, ar_tfms[self.ar_marker])
                              })
        
            
        
        if self.hydra_marker is not None:
            hyd_tfms = gmt.get_hydra_transforms(self.parent_frame, hydras=[self.hydra_marker])
            if hyd_tfms:
                tfm = hyd_tfms[self.hydra_marker]
                transforms.append({'parent':self.parent_frame,
                                   'child':'%sgripper_hydra_tooltip'%self.lr,
                                   'tfm':self.get_tooltip_transform(self.hydra_marker, tfm)
                                  })

        return transforms
