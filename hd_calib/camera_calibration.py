import rospy
import numpy as np
import cv2
import time

from hd_utils import ros_utils as ru, clouds, conversions, utils
from hd_utils.colorize import *
from hd_utils.defaults import tfm_link_rof

from cameras import RosCameras
import get_marker_transforms as gmt

asus_xtion_pro_f = 544.260779961

"""
Calibrates based on AR markers.
TODO: if three kinects work together, maybe do graph optimization.
"""


def find_rigid_tfm (points1, points2, homogeneous=True):
    """
    Gives transform from frame of points 1 to frame of points 2.
    Which means, applying the transform to points2 will make them in 
    points1' frame.
    """
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
    
    # Want to go from 1 to 2
    Y = points1 - center1
    X = points2 - center2
    
    S = X.T.dot(Y)
    # svd gives U, Sigma and V.T
    U, Sig, V = np.linalg.svd(S, full_matrices=True)

    ref_rot = np.eye(3,3)
    ref_rot[2,2] = np.round(np.linalg.det(V.T.dot(U.T)))

    R = V.T.dot(ref_rot.dot(U.T))
    t = center1 - R.dot(center2)
    
    if homogeneous:
        Tfm = np.eye(4,4)
        Tfm[0:3,0:3] = R
        Tfm[0:3,3] = t
        return Tfm
    else:
        return R,t

   
def common_ar_markers (ar_pos1, ar_pos2):
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
        #points.append(p+dist*x)
        #points.append(p+dist*y)
        #points.append(p+dist*z)
        
    return points

class CameraCalibrator:
    """
    This class uses Cyni to calibrate between cameras.
    """
    
    cameras = None
    # keeping this as fixed for now. Simple fix to change it to arbitrary frame.
    parent_frame = None
    num_cameras = 0

    camera_transforms = {}
    
    def __init__(self, cameras, parent_frame = "camera1_rgb_optical_frame"):
        
        self.cameras = cameras
        self.num_cameras = self.cameras.num_cameras
        
        assert self.num_cameras > 0
        
        self.calibrated = self.num_cameras == 1
        self.parent_frame = parent_frame
        
    def find_transform_between_cameras_from_obs (self, c1, c2):
        """
        Finds transform between cameras from latest stored observation.
        """
        if c1 > self.num_cameras or c1 < 0:
            raise Exception("Index out of range: %d"%c1)
        if c2 > self.num_cameras or c2 < 0:
            raise Exception("Index out of range: %d"%c2)
        if c1 == c2:
            return np.eye(4)

        if not self.observed_ar_transforms or \
           not self.observed_ar_transforms[c1] or \
           not self.observed_ar_transforms[c2]:
            yellowprint("Not enough information to find transform.")
            return

        ar1, ar2 = common_ar_markers(self.observed_ar_transforms[c1], self.observed_ar_transforms[c2])
        if not ar1 or not ar2:
            yellowprint("Did not find common visible AR markers between cameras %d and %d."%(c1,c2))
            return
        
        if len(ar1.keys()) == 1:
            transform = ar1.values()[0].dot(np.linalg.inv(ar2.values()[0]))
        else:
            transform = find_rigid_tfm(convert_hmats_to_points(ar1.values()),
                                       convert_hmats_to_points(ar2.values()))
            
#         import IPython
#         IPython.embed()

        return transform
    
    def extend_camera_pointsets(self, c1, c2):
        if not self.observed_ar_transforms or \
           not self.observed_ar_transforms[c1] or \
           not self.observed_ar_transforms[c2]:
            yellowprint("Not enough points found from cameras %i,%i iteration."%(c1,c2))
            return False

        ar1, ar2 = common_ar_markers(self.observed_ar_transforms[c1], self.observed_ar_transforms[c2])
        if not ar1 or not ar2:
            yellowprint("Did not find common visible AR markers between cameras %i and %i."%(c1,c2))
            return False
        
        if (c1,c2) not in self.point_list:
            self.point_list[(c1,c2)] = {c1:[],c2:[]}
            
        self.point_list[(c1,c2)][c1].extend(convert_hmats_to_points(ar1.values()))
        self.point_list[(c1,c2)][c2].extend(convert_hmats_to_points(ar2.values()))
        greenprint("Extended pointsets by %i"%(len(ar1)*4))
        
        return True
                

        
    def initialize_calibration(self):
        # Stores transforms between cameras
        #self.transform_list = {} 
        self.point_list = {}
        
    
    def process_observation(self, n_avg=5):
        """
        Get an observation and update transform list.
        """
        
        self.observed_ar_transforms = {i:{} for i in xrange(self.num_cameras)}
        
        sleeper = rospy.Rate(30)
        for i in xrange(n_avg):
            greenprint("Averaging %d out of %d"%(i+1,n_avg), False)
            for j in xrange(self.num_cameras):
                tfms = self.cameras.get_ar_markers(camera=j)
                for marker in tfms: 
                    if marker not in self.observed_ar_transforms[j]:
                        self.observed_ar_transforms[j][marker] = []
                    self.observed_ar_transforms[j][marker].append(tfms[marker])
            sleeper.sleep()

        #print self.observed_ar_transforms
        for i in self.observed_ar_transforms:
            for marker in self.observed_ar_transforms[i]:
                self.observed_ar_transforms[i][marker] = utils.avg_transform(self.observed_ar_transforms[i][marker])        

#         import IPython
#         IPython.embed()

        got_something = False
        for i in xrange(1,self.num_cameras):
            result = self.extend_camera_pointsets(0, i)
            if result is False:
                redprint("Did get info for cameras 0 and %d"%i)
                continue
            got_something = True
        
        return got_something

    def finish_calibration(self):
        """
        Average out transforms and store final values.
        Return true/false based on whether transforms were found. 
        """
        if not self.point_list: return False

        for c1,c2 in self.point_list:
            points_c1 = self.point_list[c1,c2][c1]
            points_c2 = self.point_list[c1,c2][c2]
            tfm = find_rigid_tfm(points_c1, points_c2)
            cam_transform= {}
            cam_transform['parent'] = 'camera%d_link'%(c1+1)
            cam_transform['child'] = 'camera%d_link'%(c2+1)
            cam_transform['tfm'] = tfm_link_rof.dot(tfm).dot(np.linalg.inv(tfm_link_rof))

            self.camera_transforms[c1,c2] = cam_transform
        
        self.cameras.calibrated = True
        self.cameras.store_calibrated_transforms(self.camera_transforms)    
        return True
    
    def calibrate (self, n_obs=10, n_avg=5):
        if self.num_cameras == 1:
            redprint ("Only one camera. You don't need to calibrate.", True)
            self.calibrated = True
            self.cameras.calibrated = True
            return
        
        self.initialize_calibration()
        i = 0
        while i < n_obs:
            yellowprint("Please hold still for a few seconds. Make sure the transforms look good on rviz.")
            raw_input(colorize("Observation %d from %d. Press return when ready."%(i,n_obs),'green',True))
            got_something =  self.process_observation(n_avg)
            if got_something: i += 1

        self.calibrated = self.finish_calibration()
        self.cameras.calibrated  = self.calibrated
        
    def get_transforms(self):
        if self.num_cameras == 1:
            yellowprint("Only have one camera. No transforms.")
            return
        if not self.calibrated:
            redprint("Cameras not calibrated.")
            return
        
        return self.camera_transforms.values()
        
        
    def reset_calibration (self):
        if self.num_cameras >1:
            self.calibrated = False
            self.cameras.calibrated = False
            self.camera_transforms = {}
            self.cameras.stored_tfms = {}