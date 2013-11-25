from __future__ import division

import roslib; roslib.load_manifest('tf')
import rospy, tf

import numpy as np, numpy.linalg as nlg
import scipy as scp, scipy.optimize as sco

from hd_utils import conversions, utils
from hd_utils.defaults import tfm_link_rof
from hd_utils.solve_sylvester import solve4
from hd_utils.colorize import *

import get_marker_transforms as gmt

np.set_printoptions(precision=5, suppress=True)

class HydraCalibrator:
    
    # Not dealing with some other parent frame for now. Simple addition
    parent_frame = {}
    cameras = None
    transforms = {}
    calib_hydra = None
    ar_marker = None
    calib_camera = None
    calibrated = False

    def __init__(self, cameras, ar_marker = 0, calib_hydra='left', calib_camera=None):
        """
        calib_hydra is the hydra you want to calibrate with. Either left or right.
        """
        self.cameras = cameras
        # Parent frame to get obs for camera is camera_dof
        # But, will convert to camera_link
        self.parent_frame = {'camera':'camera1_rgb_optical_frame',
                             'pr2':'base_footprint'}
        self.child_frame = {'pr2':'r_gripper_tool_tip'}
        self.calib_hydra = calib_hydra
        self.ar_marker = ar_marker
        self.calib_camera = calib_camera
    
    def set_parent_frame(self, calib_type, parent_frame):
        self.parent_frame[calib_type] = parent_frame
        
    def set_child_frame(self, calib_type='pr2', child_frame='r_gripper_tool_tip'):
        self.child_frame[calib_type] = child_frame

    def change_ar_marker(ar_marker):
        self.ar_marker = ar_marker
    
    def flip_calib_hydra (self):
        self.calib_hydra = {'left':'right', 'right':'left'}[self.calib_hydra]
    
    def get_hydra_transform(self):
        tfms = gmt.get_hydra_transforms('hydra_base', [self.calib_hydra])
        if tfms is None: return None
        return tfms[self.calib_hydra]
    
    def get_ar_transform (self):
        ar_transforms = self.cameras.get_ar_markers(markers=[self.ar_marker], camera=self.calib_camera)
        if not ar_transforms:
            return None
        return ar_transforms[self.ar_marker]
    
    def get_pr2_transform (self):
        return gmt.get_transform_frames(self.parent_frame['pr2'], self.child_frame['pr2'])

    def initialize_calibration(self, calib_type='camera'):
        
        self.calib_transforms = []
        self.hydra_transforms = []
        
        self.calib_func = {'pr2':self.get_pr2_transform,
                           'camera':self.get_ar_transform}[calib_type]
        
        assert self.cameras.calibrated or self.calib_camera is not None

    def process_observation(self, n_avg=5):
        """
        Store an observation of ar_marker and hydras.
        """        
        raw_input(colorize("Press return when ready to capture transforms.", "green", True))
        
        calib_avg_tfm = []
        hydra_avg_tfm = []
        j = 0
        thresh = n_avg*2
        
        sleeper = rospy.Rate(30)
        while j < n_avg:
            print colorize('\tGetting averaging transform : %d of %d ...'%(j,n_avg-1), "blue", True)

            calib_tfm = self.calib_func()
            hydra_tfm = self.get_hydra_transform()
            if calib_tfm is None or hydra_tfm is None:
                if calib_tfm is None:
                    yellowprint('Could not find all required AR transforms.')
                else:
                    yellowprint('Could not find all required hydra transforms.')
                thresh -= 1
                if thresh == 0: return False
                continue
                
            calib_avg_tfm.append(calib_tfm)
            hydra_avg_tfm.append(hydra_tfm)
            j += 1
            sleeper.sleep()

        self.calib_transforms.append(utils.avg_transform(calib_avg_tfm))
        self.hydra_transforms.append(utils.avg_transform(hydra_avg_tfm))
        return True
    
    def finish_calibration(self, calib_type='camera'):
        """
        Finds the final transform between parent_frame and hydra_base.
        """
        if not self.calib_transforms or not self.hydra_transforms: return False
        if len(self.calib_transforms) != len(self.hydra_transforms): return False
        
        Tas = solve4(self.calib_transforms, self.hydra_transforms)
        avg_hydra_base_transforms = [calib_tfm.dot(Tas).dot(np.linalg.inv(hydra_tfm)) for calib_tfm, hydra_tfm in zip(self.calib_transforms, self.hydra_transforms)]
        
        tfm = {}
        tfm['tfm'] = utils.avg_transform(avg_hydra_base_transforms)

        if calib_type is 'camera':
            pf = self.parent_frame[calib_type]
            cname = pf.split('_')[0]
            tfm['parent'] = cname+'_link'
            tfm['tfm'] = tfm_link_rof.dot(tfm['tfm'])
        else:
            tfm['parent'] = self.parent_frame[calib_type]

        tfm['child'] = 'hydra_base'            
        self.transforms[calib_type] = tfm 


        return True
    
    def calibrate(self, calib_type, n_obs, n_avg):
        self.initialize_calibration(calib_type)
        i = 0
        while i < n_obs:
            yellowprint ("Transform %d out of %d."%(i,n_obs))
            worked = self.process_observation(n_avg)
            if not worked:
                yellowprint("Something went wrong. Try again.")
            else:
                i += 1
        self.calibrated = self.finish_calibration(calib_type)
        
    def get_transforms(self, calib_type):
        if not self.calibrated:
            redprint("Hydras not calibrated.")
            return []

        transform = self.transforms.get(calib_type)
        if transform is None:
            redprint("Could not find calib type.")
            return []

        return [transform]

    def reset_calibration (self, calib_type=None):
        self.calibrated = False
        self.calib_transforms = []
        self.hydra_transforms = []
        if calib_type is None:
            self.transforms = {}
        elif self.transforms.get(calib_type) is not None:
            self.transforms.pop(calib_type)
