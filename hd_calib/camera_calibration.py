import rospy
import numpy as np
import cv2
import time

from hd_utils import ros_utils as ru, clouds, conversions, utils
from hd_utils.colorize import *

from cyni_cameras import cyni_cameras
import get_marker_transform as gmt

asus_xtion_pro_f = 544.260779961

"""
Calibrates based on AR markers.
"""


def find_rigid_tfm (points1, points2, homogeneous=True):
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    
    if points1.shape != points2.shape:
        yellowprint("Not the same number of points", False)
        return
    elif points1.shape[0] < 3:
        yellowprint("Not enough points", False)
        return
    
    center1 = points1.sum(axis=0)/float(points1.shape[0])
    center2 = points2.sum(axis=0)/float(points2.shape[0])
    
    X = points1 - center1
    Y = points2 - center2
    
    S = X.T.dot(Y)
    # svd gives U, Sigma and V.T
    U, Sig, V = np.linalg.svd(S, full_matrices=True)

    ref_rot = np.eye(3,3)
    ref_rot[2,2] = np.round(np.linalg.det(V.dot(U.T)))
    
#     import IPython
#     IPython.embed()   
    
    R = V.T.dot(ref_rot.dot(U.T))
    t = center2 - R.dot(center2)
    
    if homogeneous:
        Tfm = np.eye(4,4)
        Tfm[0:3,0:3] = R
        Tfm[0:3,3] = t
        return Tfm
    else:
        return R,t

   
def find_common_ar_markers (ar_pos1, ar_pos2):
    """
    Finds the common markers between two sets of markers.
    """
    ar1, ar2 = {}, {}
    common_id = np.intersect1d(ar_pos1.keys(), ar_pos2.keys())
    
    for i in common_id:
        ar1[i] = ar_pos1[i]
        ar2[i] = ar_pos2[i]
        
    return ar1, ar2

def convert_hmats_to_points (hmats):
    """
    Adds a point for the origin and one for each axis.
    Could be readily changed for something similar.
    """
    
    dist = 0.05
    points = []
    for hmat in hmats:
        x,y,z,p = hmat[0:3].T
        
        points.append(p)
        points.append(p+dist*x)
        points.append(p+dist*y)
        points.append(p+dist*z)
        
    return points

class camera_calibrator:
    """
    This class uses Cyni to calibrate between cameras.
    """
    
    cameras = None
    # keeping this as fixed for now. Simple fix to change it to arbitrary frame.
    parent_frame = None
    num_cameras = 0
    # Maybe you don't need this if acquiring information is going to take some time.
    emitter_flip_time = 0.5
    camera_transforms = {}
    
    def __init__(self, cameras, parent_frame = "camera1_depth_optical_frame"):
        
        self.cameras = cameras
        self.num_cameras = self.cameras.num_cameras
        
        assert self.num_cameras > 0
        
        self.parent_frame = parent_frame

        
    def find_transform_between_cameras_from_obs (self, c1, c2):
        """
        Finds transform between cameras from latest stored observation.
        """
        if c1 > self.num_cameras:
            raise Exception("Index out of range: %d"%c1)
        if c2 > self.num_cameras:
            raise Exception("Index out of range: %d"%c2)
        if c1 == c2:
            return np.eye(4)

        if not self.observed_ar_transforms or 
           not self.observed_ar_transforms[c1] or 
           not self.observed_ar_transforms[c2]:
            yellowprint("Not enough information to find transform.")
            return

        ar1, ar2 = common_ar_markers(self.observed_ar_transforms[c1], self.observed_ar_transforms[c2])
        if not ar1 or not ar2:
            yellowprint("Did not find common visible AR markers between cameras %d and %d."%(c1,c2))
        
        transform = rigid_tfm(convert_hmat_to_points(ar1.values()),
                              convert_hmat_to_points(ar2.values()))

        
    def initialize_calibration(self):
        if self.num_cameras == 1:
            redprint("Only one camera. You don't need to calibrate.")
            return

        # Stores transforms between cameras 
        self.transform_list = {}
        self.cameras.start_streaming()
    
    def process_observation(self, n_avg=5):
        """
        Get an observation and update transform list.
        """
        if self.num_cameras == 1:
            redprint ("Only one camera. You don't need to calibrate.", True)
            return
        
        raw_input(colorize("Press return when you're ready to take the next observation from the cameras.",'green',True))
        yellowprint("Please hold still for a few seconds.")

        self.observation_info = {i:[] for i in xrange(self.num_cameras)}
        self.observed_ar_transforms = {i:{} for i in xrange(self.num_cameras)}

        # Get RGBD observations
        for i in xrange(n_avg):
            print colorize("Transform %d out of %d for averaging."%(i,n_avg),'yellow',False)
            data = self.cameras.get_RGBD()
            for j,cam_data in data.items():
                self.observation_info[j].append(cam_data)
        
        # Find AR transforms from RGBD observations and average out transforms.
        for i in self.observation_info:
            for obs in self.observation_info[i]:
                ar_pos = gmt.get_ar_marker_poses (obs['rgb'], obs['depth'])
                for marker in ar_pos:
                    if self.observed_ar_transforms[i].get(marker) is None:
                        self.observed_ar_transforms[i][marker] = []
                    self.observed_ar_transforms[i][marker].append(ar_pos[marker])
            for marker in self.observed_ar_transforms[i]:
                self.observed_ar_transforms[i][marker] = utils.avg_transform(self.observed_ar_transforms[i][marker]) 

        for i in xrange(1:self.num_cameras)
            transform = self.find_transform_between_cameras(0, i)
            if transform is None:
                yellowprint("Did not find a transform between cameras 0 and %d"%i)
                continue
            if self.transform_list.get(0,i) is None:
               self.transform_list[0,i] = []
            self.transform_list[0,i].append(transform)

    def finish_calibration(self):
        """
        Average out transforms and store final values.
        Return true/false based on whether transforms were found. 
        """
        if not self.transform_list: return False

        for c1,c2 in self.transform_list:
            cam_transform= {}
            cam_transform['tfm'] = utils.avg_transform(self.transform_list[key])
            cam_transform['parent'] = 'camera%d_depth_optical_frame'%(c1+1)
            cam_transform['child'] = 'camera%d_depth_optical_frame'%(c2+1)
            self.camera_transforms[c1,c2] = cam_transform
            
        self.cameras.stop_streaming()
        self.cameras.store_calibrated_transforms(self.camera_transforms)
    
        return True
    
    def calibrate (self, n_obs=10, n_avg=5):
        if num_cameras == 1:
            redprint ("Only one camera. You don't need to calibrate.", True)
            return
        
        self.initialize_calibration()
        for i in range(n_obs)
            yellowprint ("Transform %d out of %d."%(i,n_obs))
            process_observation(n_avg)
        self.calibrated = finish_calibration()
        
    def get_transforms(self):
        if self.num_cameras == 1:
            yellowprint("Only have one camera. No transforms.")
            return
        if not self.calibrated:
            redprint("Cameras not calibrated.")
            return
        
        return self.camera_transforms.values()
        
        
    def reset_calibration (self):
        self.calibrated = False
        self.camera_transforms = {}
        self.cameras.stop_streaming()
        self.cameras.stored_tfms = {}