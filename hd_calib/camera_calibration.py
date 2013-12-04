import rospy
import numpy as np, numpy.linalg as nlg
import cv2, cv
import time

from hd_utils import ros_utils as ru, clouds, conversions, utils, chessboard_utils as cu
from hd_utils.colorize import *
from hd_utils.defaults import tfm_link_rof

from cameras import RosCameras
import get_marker_transforms as gmt

import roslib; roslib.load_manifest('icp_service')
from icp_service.srv import ICPTransform, ICPTransformRequest, ICPTransformResponse

asus_xtion_pro_f = 544.260779961
WIN_NAME = 'cv_test'
cb_rows = 3
cb_cols = 4
# inches
cb_size = 0.108
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
        yellowprint("Not enough points: %i"%points1.shape[0], False)
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
        points.append(p+dist*x)
        points.append(p+dist*y)
        points.append(p+dist*z)
        
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
        
        p1,p2,dist = cu.get_corresponding_points(self.observed_cb_points[c1], 
                                                 self.observed_cb_points[c2],
                                                 self.est_tfms[c1,c2])
        print "Distance difference between camera %i and camera %i points:", dist
        
        self.point_list[(c1,c2)][c1].extend(p1)
        self.point_list[(c1,c2)][c2].extend(p2)
        greenprint("Extended pointsets by %i"%(p1.shape[0]))

        
    def initialize_calibration(self, method):
        if method == 'pycb':
            self.image_list = {i:[] for i in range(self.num_cameras)}
        else:
            self.point_list = {}
        
        if method == 'cb':
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
                rtn, points = cu.get_corners_from_pc(pc)
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

    def process_observation_pycb(self):
        """
        Get an observation and update transform list.
        """
        sleeper = rospy.Rate(10)
        for j in xrange(self.num_cameras):
            pc = self.cameras.get_pointcloud(j)
            xyz, rgb = ru.pc2xyzrgb(pc)
            rgb = np.copy(rgb)
            _, corners = cu.get_corners_rgb(rgb,method='cv', rows=cb_rows, cols=cb_cols)
            if len(corners) < cb_cols*cb_rows*1.0/3.0:
                redprint ("Found too few corners: %i"%len(corners)) 
                return False
            self.image_list[j].append(rgb)

        return True
    
    def finish_calibration_pycb(self, avg=False):
        
        cb_transforms = {i:[] for i in range(self.num_cameras)}
        I = np.eye(4)
        
        for i in self.image_list:
            yellowprint("Getting transform data for camera %i."%(i+1))
            ind = 1
            for img in self.image_list[i]:
                blueprint("... Observation %i."%ind)
                ind += 1

                tfm = cu.get_checkerboard_transform(img, cb_size=cb_size)
                if tfm is None: return False
                cb_transforms[i].append(tfm)
        
        rel_tfms = {i:[] for i in range(1,self.num_cameras)}
        for i in range(1,self.num_cameras):
            cam_transform= {}
            cam_transform['parent'] = 'camera1_link'
            cam_transform['child'] = 'camera%d_link'%(i+1)
            for j in range(len(cb_transforms[i])):
                rtfm = cb_transforms[0][j].dot(nlg.inv(cb_transforms[i][j]))
                rel_tfms[i].append(rtfm)
            if avg:
                tfm = utils.avg_transform(rel_tfms[i])
                # convert from in to cm
                tfm[0:3,3] *= 1.0#0.0254
                cam_transform['tfm'] = tfm_link_rof.dot(tfm).dot(np.linalg.inv(tfm_link_rof))

            else:
                scores = []
                for j,rtfm in enumerate(rel_tfms[i]):
                    scores.append(0)
                    rtfm = rel_tfms[i][j]
                    for k,rtfm2 in enumerate(rel_tfms[i]):
                        if j == k:
                            continue
                        scores[j] += nlg.norm(rtfm.dot(nlg.inv(rtfm2))-I)
                print "Scores:", scores
                tfm = rel_tfms[i][np.argmin(scores)]
                # convert from in to m
                tfm[0:3,3] *= 1.0#0.0254
                cam_transform['tfm'] = tfm_link_rof.dot(tfm).dot(np.linalg.inv(tfm_link_rof))
            self.camera_transforms[0,i] = cam_transform
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
    
    def calibrate (self, method='ar', use_icp=False, n_obs=10, n_avg=5):
        if self.num_cameras == 1:
            redprint ("Only one camera. You don't need to calibrate.", True)
            self.calibrated = True
            self.cameras.calibrated = True
            return
        
        self.initialize_calibration(method)
        i = 0
        while i < n_obs:
            yellowprint("Please hold still for a few seconds. Make sure the transforms look good on rviz.")
            raw_input(colorize("Observation %d from %d. Press return when ready."%(i+1,n_obs),'green',True))
            if method=='ar':
                got_something =  self.process_observation_ar(n_avg)
            elif method=='cb':
                got_something =  self.process_observation_cb()
            else:
                got_something =  self.process_observation_pycb()
            if got_something: i += 1

        if method=='pycb':
            self.calibrated = self.finish_calibration_pycb()
        else:
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