import numpy as np
import rospy
import numpy.linalg as nlg
import get_marker_transforms as gmt

from hd_utils.colorize import *
from hd_utils import utils  

class Gripper:
    """
    Old version of gripper, assumes gripper has more than one markers (in fact three markers?)
    """
    
    # Change if need be
    parent_frame = 'camera1_rgb_optical_frame'
    
    cameras = None
    
    transform_graph = None
    tooltip_calculated = False
    lr = None
    mmarkers = []
    lmarkers = []
    rmarkers = []
    allmarkers = []
    
    ar_markers = []
    hydra_markers = []
    
    def __init__ (self, lr, transform_graph, cameras=None):
        self.transform_graph = transform_graph
        self.cameras = cameras
        if self.cameras is not None:
            self.parent_frame = self.cameras.parent_frame

        for group in self.transform_graph.nodes_iter():
            self.ar_markers.extend(self.transform_graph.node[group]['ar_markers'])
            self.hydra_markers.extend(self.transform_graph.node[group]['hydras'])

        self.mmarkers = self.transform_graph.node['master']['graph'].nodes()        
        self.lmarkers = self.transform_graph.node['l']['graph'].nodes()
        self.rmarkers = self.transform_graph.node['r']['graph'].nodes()
        self.allmarkers = self.mmarkers + self.rmarkers + self.lmarkers
        self.lr = lr
        
    def set_cameras (self, cameras):
        self.cameras = cameras
        
    def get_ar_markers(self):
        return self.ar_markers
    
    def calculate_tooltip_transform (self, m1, m2):
        """
        Not used in any places
        Assuming that the gripper has "master", "l" and "r"
        m1 and m2 are markers on the tool tip. 
        """
        if self.tooltip_calculated:
            yellowpring("Tool tip already calculated.")
            return
        if m1 in self.rmarkers:
            assert m2 in self.lmarkers
            m1, m2 = m2, m1
        else:
            assert m1 in self.lmarkers
            assert m2 in self.rmarkers

        # Assume that the origin of the tool tip transform is the avg
        # point of fingers when they're at an angle of 0
        ltfm = self.transform_graph.edge['master']['l']['tfm_func'].get_tfm('cor', m1, 0)
        rtfm = self.transform_graph.edge['master']['r']['tfm_func'].get_tfm('cor', m2, 0)
        
        lorg = ltfm[0:3, 3]
        rorg = ltfm[0:3, 3]
        tt_org = (lorg + rorg) / 2.0

        master = self.transform_graph.node['master']['graph']
        
        # Old "correct" code.
        # z axis is the same as cor
        # tt_z = np.array([0,0,1])
        # x axis (pointing axis) is the projection of vector from cor to tt_org on the xy plane.
        # Since we're in the 'cor' frame, it's just the projection of direction vector tt_org on xy plane.
        # tt_org_vec = tt_org/nlg.norm(tt_org)
        # tt_x = tt_org_vec - tt_org_vec.dot(tt_z)*tt_z
        # tt_x = tt_x/nlg.norm(tt_x)
        # tt_y = np.cross(tt_z, tt_x)
        # tt_tfm =  np.r_[np.c_[tt_x, tt_y, tt_z, tt_org], np.array([[0,0,0,1]])]
        

        # New hacky code:
        master_marker = self.transform_graph.node['master']['master_marker']
        master_tfm = master.edge['cor'][master_marker]['tfm']
        tt_tfm = np.r_[np.c_[master_tfm[0:3, 0:3], tt_org], np.array([[0, 0, 0, 1]])]
        
        
        master.add_node('tool_tip')
        master.add_edge('cor', 'tool_tip')
        master.add_edge('tool_tip', 'cor')
        
        master.edge['cor']['tool_tip']['tfm'] = tt_tfm
        master.edge['tool_tip']['cor']['tfm'] = nlg.inv(tt_tfm)
        
        for node in master.nodes_iter():
            if node == 'cor' or node == 'tool_tip': continue
            master.add_edge(node, 'tool_tip')
            master.add_edge('tool_tip', node)
            
            tfm = master.edge[node]['cor']['tfm'].dot(tt_tfm)
        
            master.edge[node]['tool_tip']['tfm'] = tfm
            master.edge['tool_tip'][node]['tfm'] = nlg.inv(tfm)
        
        self.mmarkers.append('tool_tip')
        self.allmarkers.append('tool_tip')
        self.tooltip_calculated = True

        
    def get_tooltip_transform(self, marker_tfms, theta):
        """
        Get tool tip transform from dict of relevant markers to transforms
        Also provide the angle of gripper.
        Don't need it if only master transform is visible.
        m has to be on one of the 'master,'l' or 'r' groups
        """
        if self.tooltip_calculated is False:
            redprint ("Tool tip transform not calibrated.")
            return

        masterg = self.transform_graph.node['master']['graph']

        avg_tfms = []
        
        for m in marker_tfms:
            if m in self.mmarkers:
                tt_tfm = masterg.edge[m]['tool_tip']['tfm']
            elif m in self.lmarkers:
                tt_tfm = self.transform_graph.edge['l']['master']['tfm_func'].get_tfm(m, 'tool_tip', theta)
            elif m in self.rmarkers:
                tt_tfm = self.transform_graph.edge['r']['master']['tfm_func'].get_tfm(m, 'tool_tip', theta)
            else:
                redprint('Marker %s not on gripper.' % m)
                continue
            avg_tfms.append(marker_tfms[m].dot(tt_tfm))
            
        if len(avg_tfms) == 0:
            redprint('No markers on gripper found.')
            return

        return utils.avg_transform(avg_tfms)
    
    def get_rel_transform(self, m1, m2, theta):
        """
        Return relative transform between any two markers on gripper (from m2 to m1)
        """
        masterg = self.transform_graph.node['master']['graph']

        if m1 in self.mmarkers:
            tfm1 = masterg.edge[m1]['cor']['tfm']
        elif m1 in self.lmarkers:
            tfm1 = self.transform_graph.edge['l']['master']['tfm_func'].get_tfm(m1, 'cor', theta)
        elif m1 in self.rmarkers:
            tfm1 = self.transform_graph.edge['r']['master']['tfm_func'].get_tfm(m1, 'cor', theta)
        else:
            redprint('Marker %s not on gripper.' % m1)
            return

        if m2 in self.mmarkers:
            tfm2 = masterg.edge['cor'][m2]['tfm']
        elif m2 in self.lmarkers:
            tfm2 = self.transform_graph.edge['master']['l']['tfm_func'].get_tfm('cor', m2, theta)
        elif m2 in self.rmarkers:
            tfm2 = self.transform_graph.edge['master']['r']['tfm_func'].get_tfm('cor', m2, theta)
        else:
            redprint('Marker %s not on gripper.' % m2)
            return
        
        return tfm1.dot(tfm2)
    
    def get_all_transforms(self, parent_frame, diff_cam=False):
        """
        From marker transforms given in parent_frame, get all transforms
        of all markers on gripper.
        """
    	if self.cameras is None:
    	    redprint("Cameras not initialized. Could not get transforms.")
    	    return

        ar_tfms = self.cameras.get_ar_markers()

        if diff_cam:
            ar_tfms_cam = {}
            for i in range(self.cameras.num_cameras):
                ar_tfms_cam[i] = self.cameras.get_ar_markers(camera=i, parent_frame=True)

        hyd_tfms = gmt.get_hydra_transforms(self.parent_frame, None)
        theta = gmt.get_pot_angle(self.lr)

        marker_tfms = ar_tfms
        marker_tfms.update(hyd_tfms)
        
        ret_tfms = []
        
        cor_avg_tfms = []
        cor_hyd_avg = []
        cor_ar_avg = []
        ar_found = False
        hyd_found = False
        for m, tfm in marker_tfms.items():
            if m in self.ar_markers:
                c_tfm = tfm.dot(self.get_rel_transform(m, 'cor', theta)) 
                cor_ar_avg.append(c_tfm)
                cor_avg_tfms.append(c_tfm)
                ar_found = True
            elif m in self.hydra_markers:
                c_tfm = tfm.dot(self.get_rel_transform(m, 'cor', theta)) 
                cor_hyd_avg.append(c_tfm)
                cor_avg_tfms.append(c_tfm)
                hyd_found = True
        
        if diff_cam:
            cor_ar_cams = {}
            cam_found = {i:False for i in ar_tfms_cam}
            for i in ar_tfms_cam:
                for m, tfm in ar_tfms_cam[i].items():
                    if m in self.ar_markers:
                        c_tfm = tfm.dot(self.get_rel_transform(m, 'cor', theta))
                        if i not in cor_ar_cams:
                            cor_ar_cams[i] = []
                            cam_found[i] = True 
                        cor_ar_cams[i].append(c_tfm)
                if cam_found[i]:
                    cor_ar_cams[i] = utils.avg_transform(cor_ar_cams[i])


        if len(cor_avg_tfms) == 0: return ret_tfms
        
        cor_tfm = utils.avg_transform(cor_avg_tfms)
        cor_h_tfm = utils.avg_transform(cor_hyd_avg)
        cor_a_tfm = utils.avg_transform(cor_ar_avg)
        
        ret_tfms.append({'parent':parent_frame,
                         'child':'%sgripper_%s' % (self.lr, 'cor'),
                         'tfm':cor_tfm})

        for m in self.allmarkers:
            if m != 'cor' and m != 'tool_tip':
                tfm = self.get_rel_transform('cor', m, theta)
                ret_tfms.append({'parent':parent_frame,
                                 'child':'%sgripper_%s' % (self.lr, m),
                                 'tfm':cor_tfm.dot(tfm)})

        if self.tooltip_calculated:
            tfm = self.get_rel_transform('cor', 'tool_tip', theta)
            ret_tfms.append({'parent':parent_frame,
                             'child':'%sgripper_tooltip' % self.lr,
                             'tfm':cor_tfm.dot(tfm)})
            if hyd_found:
                ret_tfms.append({'parent':parent_frame,
                                 'child':'%sgripper_tooltip_hydra' % self.lr,
                                 'tfm':cor_h_tfm.dot(tfm)})
            if ar_found:
                ret_tfms.append({'parent':parent_frame,
                                 'child':'%sgripper_tooltip_ar' % self.lr,
                                 'tfm':cor_a_tfm.dot(tfm)})
            
            if diff_cam:
                for i, cor_tfm in cor_ar_cams.items():
                    ret_tfms.append({'parent':parent_frame,
                                     'child':'%sgripper_tooltip_camera%i' % (self.lr, i + 1),
                                     'tfm':cor_tfm.dot(tfm)})
                    
        
        return ret_tfms
    
    
    def get_specific_tooltip_tfm (self, m_type='AR'):
        """
        Not used in any places
        Get the estimate of the tool tip from either ar_markers or hydras.
        """
        if m_type not in ['AR', 'hydra']:
            redprint('Not sure what estimate %s gives.' % m_type)
            return
        
        if m_type == 'AR':
            if len(self.ar_markers) == 0:
                redprint('No AR markers to give you an estimate.')
                return
            marker_tfms = self.cameras.get_ar_markers(markers=self.ar_markers)
        else:
            if len(self.hydra_markers) == 0:
                redprint('No hydras to give you an estimate.')
                return
            marker_tfms = gmt.get_hydra_transforms(self.parent_frame, self.hydra_markers)
        
        theta = gmt.get_pot_angle(self.lr)
        return self.get_tooltip_transform(marker_tfms, theta)
        
    
    def get_markers_transform (self, markers, marker_tfms, theta):
        """
        Takes in marker_tfms found, angles and markers for
        which transforms are required.
        Returns a dict of marker to transforms for the markers
        requested, in the same frame as the marker tfms received.
        
        """
        
        rtn_tfms = {}
        cor_avg_tfms = []
        for m, tfm in marker_tfms.items():
            if m in self.allmarkers:
                cor_avg_tfms.append(tfm.dot(self.get_rel_transform(m, 'cor', theta)))
        
        cor_tfm = utils.avg_transform(cor_avg_tfms)

        for marker in markers:
            if marker in marker_tfms:
                rtn_tfms[marker] = marker_tfms[marker]
                continue
            
            tfm = self.get_rel_transform('cor', marker, theta)
            if tfm is not None:
                rtn_tfms[marker] = cor_tfm.dot(tfm)
        
        return rtn_tfms

        
    
    def get_obs_new_marker(self, marker, n_tfm=10, n_avg=10):
        """
        Store a bunch of readings seen for new marker being added.
        Returns a list of dicts each of which has marker transforms found
        and potentiometer angles.
        """
        
        all_obs = []
        
        i = 0
        thresh = n_avg * 2
        sleeper = rospy.Rate(30)
        
        while i < n_tfm:
            raw_input(colorize("Getting transform %i out of %i. Hit return when ready." % (i, n_tfm), 'yellow', True))
            
            j = 0
            j_th = 0
            pot_angle = 0
            found = True
            avg_tfms = []
            while j < n_avg:
                blueprint("Averaging transform %i out of %i." % (j, n_avg))
                ar_tfms = self.cameras.get_ar_markers();
                hyd_tfms = gmt.get_hydra_transforms(parent_frame=self.parent_frame, hydras=None);
                curr_angle = gmt.get_pot_angle(self.lr)
                
                if not ar_tfms and not hyd_tfms:
                    yellowprint("Could not find any transform.")
                    j_th += 1
                    if j_th < thresh:
                        continue
                    else:
                        found = False
                        break
                                
                tfms = ar_tfms
                pot_angle += curr_angle
                tfms.update(hyd_tfms)

                avg_tfms.append(tfms)
                j += 1
                sleeper.sleep()
            
            if found is False:
                yellowprint("Something went wrong; try again.")
                continue
            
            tfms_found = {}
            for tfms in avg_tfms:
                for m in tfms:
                    if m not in tfms_found:
                        tfms_found[m] = []
                    tfms_found[m].append(tfms[m])

            if marker not in tfms_found:
                yellowprint("Could not find marker to be added; try again.")
                continue
            
            for m in tfms_found:
                tfms_found[m] = utils.avg_transform(tfms_found[m])
            pot_angle = pot_angle / n_avg
            
            all_obs.append({'tfms':tfms_found, 'pot_angle':pot_angle})
            i += 1
            
        return all_obs
            

    
    def add_marker (self, marker, group, m_type='AR', n_tfm=10, n_avg=30):
        """
        Add marker to the gripper, after calibration.
        
        @marker: marker id (a number for AR markers, either 'left' or 'right' for hydras)
        @group: 'l','r' or 'master'
        @m_type: type of marker - 'AR' or 'hydra'
        @n_tfm,@n_avg - the usual 
        """
        # Checks to see if data + situation is valid to add marker
        if marker in self.allmarkers:
            greenprint("Marker already added.")
            return

        if self.cameras is None:
            redprint("Sorry, have no cameras to check.")
            return
        
        if group not in ['master', 'l', 'r']:
            redprint('Invalid group %s' % group)
            return
        
        if m_type not in ['AR', 'hydra']:
            redprint('Invalid marker type %s' % m_type)
            return

        # Get observations 
        all_obs = self.get_obs_new_marker(marker, n_tfm, n_avg)
        
        # Compute average relative transform between correct primary node and marker
        primary_node = self.transform_graph.node[group]['primary']
        avg_rel_tfms = []
        
        for obs in all_obs:
            tfms = obs['tfms']
            angle = obs['pot_angle']
            
            marker_tfm = tfms[marker]
            primary_tfm = self.get_markers_transform([primary_node], tfms, angle)[primary_node]
            
            avg_rel_tfms.append(nlg.inv(primary_tfm).dot(marker_tfm))
            
        rel_tfm = utils.avg_transform(avg_rel_tfms)
        
        # Add markers to relevant lists
        self.transform_graph.node[group]['markers'].append(marker)
        self.allmarkers.append(marker)
        
        if m_type == 'AR':
            self.transform_graph.node[group]['ar_markers'].append(marker)
            self.ar_markers.append(marker)
        else:
            self.transform_graph.node[group]['hydras'].append(marker)
            self.hydra_markers.append(marker)
            
        if group == 'master':
            self.mmarkers.append(marker)
        elif group == 'l':
            self.lmarkers.append(marker)
        else:
            self.rmarkers.append(marker)

        # Update graphs from transform found
        graph = self.transform_graph.node[group]['graph']
        graph.add_node(marker)
        
        for node in graph.nodes_iter():
            graph.add_edge(node, marker)
            graph.add_edge(marker, node)
            
            tfm = graph.edge[node][primary_node]['tfm'].dot(rel_tfm)
            
            graph.edge[node][marker]['tfm'] = tfm
            graph.edge[marker][node]['tfm'] = nlg.inv(tfm)
        
        greenprint("Successfully added your marker to the gripper!")
