import rospy
import numpy as np, numpy.linalg as nlg
import cv2, cv
import time

from hd_utils import ros_utils as ru, clouds, conversions, utils
from hd_utils.colorize import *
from hd_utils.defaults import tfm_link_rof

from cameras import RosCameras
import get_marker_transforms as gmt

import roslib; roslib.load_manifest('icp_service')
from icp_service.srv import ICPTransform, ICPTransformRequest, ICPTransformResponse

asus_xtion_pro_f = 544.260779961
WIN_NAME = 'cv_test'
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
    

cb_rows = 8
cb_cols = 6


def get_corners_rgb(rgb,rows=None,cols=None):
    cv_rgb = cv.fromarray(rgb)
    
    if not rows: rows = cb_rows
    if not cols: cols = cb_cols
    
    rtn, corners = cv.FindChessboardCorners(cv_rgb, (cb_rows, cb_cols))
    return rtn, corners

def get_xyz_from_corners (corners, xyz):
    points = []
    for j,i in corners:
        x = i - np.floor(i)
        y = j - np.floor(j)
        p1 = xyz[np.floor(i),np.floor(j)]
        p2 = xyz[np.floor(i),np.ceil(j)]
        p3 = xyz[np.ceil(i),np.ceil(j)]
        p4 = xyz[np.ceil(i),np.floor(j)]        
        p = p1*(1-x)*(1-y) + p2*(1-x)*y + p3*x*y + p4*x*(1-y)
        if np.isnan(p).any(): print p
        points.append(p)

    return np.asarray(points)

def get_corners_from_pc(pc,rows=None,cols=None):
    xyz, rgb = ru.pc2xyzrgb(pc)
    rgb = np.copy(rgb)
    rtn, corners = get_corners_rgb(rgb, rows, cols)
    if len(corners) == 0:
        return 0, None
    points = get_xyz_from_corners(corners, xyz)
    return rtn, points
    
def get_corresponding_points(points1, points2, guess_tfm, rows=None, cols=None):
    """
    Returns two lists of points such that the transform explains the relation between
    pointsets the most. Also, returns the norm of the difference between point sets.
    tfm is from cam1 -> cam2
    """
    if not rows: rows = cb_rows
    if not cols: cols = cb_cols

    
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    
    p12 = np.c_[points1,points2]
    p12 = p12[np.bitwise_not(np.isnan(p12).any(axis=1)),:]
    p1 = p12[:,0:3]
    p2 = p12[:,3:6]
    est = np.c_[p2,np.ones((p2.shape[0],1))].dot(guess_tfm.T)[:,0:3]
    dist = nlg.norm(p1-est,ord=np.inf)
    
    corr = range(rows*cols-1,-1,-1)
    p12r = np.c_[points1,points2[corr,:]]
    p12r = p12r[np.bitwise_not(np.isnan(p12r).any(axis=1)),:]
    p1r = p12r[:,0:3]
    p2r = p12r[:,3:6]
    est = np.c_[p2r,np.ones((p2r.shape[0],1))].dot(guess_tfm.T)[:,0:3]
    dist_new = nlg.norm(p1r-est, ord=np.inf)
    if dist_new < dist:
        points1, points2, dist = p1, p2, dist_new
    else:
        points1, points2 = p1, p2

    return points1, points2, dist

   
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
    
    icpService = None

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
    
    def extend_camera_pointsets_ar(self, c1, c2):
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
                
    def extend_camera_pointsets_cb(self, c1, c2):
        
        if (c1,c2) not in self.point_list:
            self.point_list[(c1,c2)] = {c1:[],c2:[]}
        
        p1,p2,dist = get_corresponding_points(self.observed_cb_points[c1], 
                                              self.observed_cb_points[c2],
                                              self.est_tfms[c1,c2])
        print "Distance difference between camera %i and camera %i points:", dist
        
        self.point_list[(c1,c2)][c1].extend(p1)
        self.point_list[(c1,c2)][c2].extend(p2)
        greenprint("Extended pointsets by %i"%(p1.shape[0]))

        
    def initialize_calibration(self, use_ar):
        self.point_list = {}
        
        if not use_ar:
            ready = False
            sleeper = rospy.Rate(10)
            while not ready:
                ready = self.estimate_initial_transform()
                sleeper.sleep()
        
    
    def process_observation_ar(self, n_avg=5):
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
            result = self.extend_camera_pointsets_ar(0, i)
            if result is False:
                redprint("Did get info for cameras 0 and %d"%i)
                continue
            got_something = True
        
        return got_something
    
    def estimate_initial_transform(self):
        raw_input("Place AR marker(s) visible to all cameras to get an initial estimate. Then hit enter.")
        N_AVG = 10

        self.est_tfms = {i:{} for i in xrange(1,self.num_cameras)}
        tfms_found = {i:{} for i in xrange(self.num_cameras)}
        sleeper = rospy.Rate(30)
        for _ in xrange(N_AVG):
            for i in xrange(self.num_cameras):
                mtfms = self.cameras.get_ar_markers(camera=i)
                for m in mtfms:
                    if m not in tfms_found[i]:
                        tfms_found[i][m] = []
                    tfms_found[i][m].append(mtfms[m])
            sleeper.sleep()
            
        for i in tfms_found:
            for m in tfms_found[i]:
                tfms_found[i][m] = utils.avg_transform(tfms_found[i][m])

        for i in xrange(1,self.num_cameras):
            ar1, ar2 = common_ar_markers(tfms_found[0], tfms_found[i])
            if not ar1 or not ar2:
                redprint("No common AR Markers found between camera 1 and %i"%(i+1))
                return False

            if len(ar1.keys()) == 1:
                self.est_tfms[0,i] = ar1.values()[0].dot(nlg.inv(ar2.values()[0]))
            else:
                self.est_tfms[0,i] = find_rigid_tfm(convert_hmats_to_points(ar1.values()),
                                           convert_hmats_to_points(ar2.values()))
        
        return True
                
    
    def process_observation_cb(self):
        """
        Get an observation and update transform list.
        """
        
        self.observed_cb_points = {}
        
        sleeper = rospy.Rate(10)
        for j in xrange(self.num_cameras):
            tries = 10
            while tries > 0:
                
                pc = self.cameras.get_pointcloud(j)
                rtn, points = get_corners_from_pc(pc)
                if rtn == 0:
                    yellowprint("Could not find all the points on the checkerboard for camera%i"%(j+1))
                    tries -= 1
                    sleeper.sleep()
                else: break
            if tries == 0:
                redprint ("Could not find all the chessboard points for camera %i."%(j+1))
                return False
            self.observed_cb_points[j] = points

        for i in xrange(1,self.num_cameras):
            self.extend_camera_pointsets_cb(0, i)
        
        return True

    def finish_calibration(self, use_icp):
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

            if use_icp:
                if self.icpService is None:
                    self.icpService = rospy.ServiceProxy("icpTransform", ICPTransform)
                
                
                greenprint("Refining calibration with ICP.")
                req = ICPTransformRequest()
                
                
                # Interchange pc1 and pc2 or use inv(cam_transform) as guess.
                raw_input(colorize("Cover camera %i and hit enter!"%(c2+1),'yellow',True))
                pc2 = self.cameras.get_pointcloud(c1)
                pc2_points = ru.pc2xyz(pc2)
                pc2_points = np.reshape(pc2_points, (640*480,3), order='F')
                pc2_points = pc2_points[np.bitwise_not(np.isnan(pc2_points).any(axis=1)),:]
                req.pc2 = ru.xyz2pc(pc2_points, pc2.header.frame_id)
    
                raw_input(colorize("Cover camera %i and hit enter!"%(c1+1),'yellow',True))
                pc1 = self.cameras.get_pointcloud(c2)
                pc1_points = ru.pc2xyz(pc1)
                pc1_points = np.reshape(pc1_points, (640*480,3), order='F')
                pc1_points = pc1_points[np.bitwise_not(np.isnan(pc1_points).any(axis=1)),:]
                pc1_points = (np.c_[pc1_points, np.ones((pc1_points.shape[0],1))].dot(tfm.T))[:,0:3]
                req.pc1 = ru.xyz2pc(pc1_points, pc1.header.frame_id)
    
                req.guess = conversions.hmat_to_pose(np.eye(4))
    
                try:
                    res = self.icpService(req)
                    print res
                    res_tfm = conversions.pose_to_hmat(res.pose)
                    ttt = tfm_link_rof.dot(res_tfm.dot(tfm)).dot(np.linalg.inv(tfm_link_rof))
                    cam_transform['tfm'] = ttt
                except:
                    redprint("ICP failed. Using AR-only calibration.")
                
            self.camera_transforms[c1,c2] = cam_transform
        
        self.cameras.calibrated = True
        self.cameras.store_calibrated_transforms(self.camera_transforms)
        return True
    
    def calibrate (self, use_ar=False, use_icp=True, n_obs=10, n_avg=5):
        if self.num_cameras == 1:
            redprint ("Only one camera. You don't need to calibrate.", True)
            self.calibrated = True
            self.cameras.calibrated = True
            return
        
        self.initialize_calibration(use_ar)
        i = 0
        while i < n_obs:
            yellowprint("Please hold still for a few seconds. Make sure the transforms look good on rviz.")
            raw_input(colorize("Observation %d from %d. Press return when ready."%(i,n_obs),'green',True))
            if use_ar:
                got_something =  self.process_observation_ar(n_avg)
            else:
                got_something =  self.process_observation_cb()
            if got_something: i += 1

        self.calibrated = self.finish_calibration(use_icp)
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