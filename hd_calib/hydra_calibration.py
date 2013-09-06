from __future__ import division

import roslib; roslib.load_manifest('tf')
import rospy, tf

import numpy as np, numpy.linalg as nlg
import scipy as scp, scipy.optimize as sco

from hd_utils import conversions, utils
from hd_utils.solve_sylvester import solve4
from hd_utils.colorize import *

import get_marker_transforms as gmt

np.set_printoptions(precision=5, suppress=True)

class hydra_calibrator:
    
    # Not dealing with some other parent frame for now. Simple addition
    parent_frame = None
    cameras = None
    hydra_base_transform = None
    calib_hydra = None
    ar_marker = None
    calib_camera = None
    calibrated = False

    def __init__(self, cameras, ar_marker = 0, parent_frame = 'camera1_depth_optical_frame', calib_hydra='left', calib_camera=None):
        """
        calib_hydra is the hydra you want to calibrate with. Either left or right.
        """
        self.cameras = cameras
        self.parent_frame = parent_frame
        self.calib_hydra = calib_hydra
        self.ar_marker = ar_marker 
        self.calib_camera = calib_camera
    
    def change_ar_marker(ar_marker):
        self.ar_marker = ar_marker
    
    def flip_calib_hydra (self):
        self.calib_hydra = {'left':'right', 'right':'left'}[self.calib_hydra]
        
    def initialize_calibration(self):
        self.ar_transforms = []
        self.hydra_transforms = []
        assert self.cameras.calibrated
        self.cameras.start_streaming()
        

    def process_observation(self, n_avg=5):
        """
        Store an observation of ar_marker and hydras.
        """
        
        raw_input(colorize("Press return when ready to capture transforms.", "green", True))
        
        ar_avg_tfm = []
        hydra_avg_tfm = []
        for j in xrange(n_avg):
            print colorize('\tGetting averaging transform : %d of %d ...'%(j,n_avg-1), "blue", True)

            ar_transforms = gmt.get_ar_markers_from_cameras(self.cameras, cams = [self.calib_camera], markers = [self.ar_marker])
            hydra_transforms = gmt.get_hydra_transforms('hydra_base', [self.calib_hydra])
            ar_avg_tfm.append(ar_transforms[self.ar_marker])
            hydra_avg_tfm.append(hydra_transforms[self.ar_marker][self.calib_hydra])
            
        self.ar_transforms.append(utils.avg_transform(ar_avg_tfm))
        self.hydra_transforms.append(utils.avg_transform(hydra_avg_tfm))
    
    def finish_calibration(self):
        """
        Finds the final transform between parent_frame and hydra_base.
        """
        if not self.ar_transforms or self.hydra_transforms: return False
        if len(self.ar_transforms) != len(self.hydra_transforms): return False
        
        Tas = solve4(self.ar_transforms, self.hydra_transforms)
        
        avg_hydra_base_transforms = [ar_tfm.dot(Tas).dot(hydra_tfm) for ar_tfm, hydra_tfm in zip(self.ar_transforms, self.hydra_transforms)]
        self.hydra_base_transform = utils.avg_transform(avg_hydra_base_transforms)
        
        self.cameras.stop_streaming()
        return True
    
    def calibrate(self, n_obs, n_avg):
        self.initialize_calibration()
        for i in range(n_obs):
            yellowprint ("Transform %d out of %d."%(i,n_obs))
            self.process_observation(n_avg)
        self.calibrated = self.finish_calibration()
        
    def get_transforms(self):
        if not self.calibrated:
            redprint("Hydras not calibrated.")
            return
        
        transform = {'parent':self.parent_frame,
                     'child': 'hydra_base',
                     'tfm': self.hydra_base_transform}
        
        return [transform]

    def reset_calibration (self):
        self.calibrated = False
        self.ar_transforms = []
        self.hydra_transforms = []
        self.cameras.stop_streaming()
