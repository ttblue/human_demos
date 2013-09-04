import rospy
import numpy as np
import cv2
import time

from hd_utils import ros_utils as ru, clouds, conversions, utils
from hd_utils.colorize import *

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


def get_ar_marker_poses (rgb, depth):
    """
    In order to run this, ar_marker_service needs to be running.
    """
    if rospy.get_name() == '/unnamed':
        rospy.init_node('keypoints')
    
    getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)
    
    #xyz = svi.transform_pointclouds(depth, tfm)
    xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
    pc = ru.xyzrgb2pc(xyz, rgb, '/base_footprint')
    
    req = MarkerPositionsRequest()
    req.pc = pc
    
    marker_tfm = {}
    res = getMarkers(req)
    for marker in res.markers.markers:
        marker_tfm[marker.id] = conversions.pose_to_hmat(marker.pose.pose)
    
    #print "Marker ids found: ", marker_tfm.keys()
    
    return marker_tfm
   
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
    This calibrator uses Cyni to calibrate between cameras.
    """
    
    devices = {}
    streams = {}
    num_cameras = 0
    # Maybe you don't need this if acquiring information is going to take some time.
    emitter_flip_time = 0.5
    camera_transforms = []
    
    def __init__(self, num_cameras=2):
        
        assert num_cameras > 0
        self.num_cameras = num_cameras
        self.parent_frame = "camera1_depth_optical_frame"
        
        cyni.initialize()
        
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
        self.allDevices = cyni.enumerateDevices()
        
        if len(self.allDevices) == 0:
            raise Exception("No devices found! Cyni not initialized properly or, devices actually not present.")
        if num_cameras > len(self.allDevices):
            redprint("Warning: Requesting more devices than available. Getting all devices.")
        
        self.num_cameras = min(num_markers,len(self.allDevices))
        
        for i in xrange(self.num_cameras):
            device = cyni.Device(self.allDevices[i]['uri'])
            device.open()
            
            self.devices[i] = device
            self.streams[i] = {}
            self.stream[i]['color'] = device.createStream("depth", width=640, height=480, fps=30)
            self.stream[i]['depth'] = device.createStream("color", width=640, height=480, fps=30)
            device.setImageRegistrationMode("depth_to_color")
            device.setDepthColorSyncEnabled(on=True)

        if self.num_cameras == 1:
            redprint("Only one camera. You don't need to calibrate.")
            return

        for i in streams:
            streams[i]['color'].start()
            streams[i]['depth'].start()
            streams[i]['depth'].setEmitterState(False)

        # Stores transforms between cameras 
        self.transform_list = {}
    
    def process_observation(self, n_avg=5):
        """
        Get an observation and update transform list.
        """
        if self.num_cameras == 1:
            redprint ("Only one camera. You don't need to calibrate.", True)
            return
        
        raw_input(colorize("Press return when you're ready to take the next observation from the cameras.",'green',True))
        yellowprint("Please hold still for a few seconds.")

        self.observation_info = {i:[] for i in streams}
        self.observed_ar_transforms = {i:{} for i in streams}

        # Get RGBD observations
        for i in xrange(n_avg):
            print colorize("Transform %d out of %d for averaging."%(i,n_avg),'yellow',False)
            for j,stream in streams.items():
                stream["depth"].setEmitterState(True)
                depth = stream["depth"].readFrame().data
                rgb = cv2.cvtColor(stream["color"].readFrame().data, cv2.COLOR_RGB2BGR)
                self.observation_info[j].append({'rgb':rgb, 'depth':depth})
                time.sleep(self.emitter_flip_time)
                stream["depth"].setEmitterState(False)
        
        # Find AR transforms from RGBD observations and average out transforms.
        for i in self.observation_info:
            for obs in self.observation_info[i]:
                ar_pos = get_ar_marker_poses (obs['rgb'], obs['depth'])
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
            cam_transform['parent'] = 'camera%d_depth_optical_frame'%c1
            cam_transform['child'] = 'camera%d_depth_optical_frame'%c2
            self.camera_transforms.append(cam_transform)
    
        return True
    
    def calibrate (self, n_obs=10, n_avg=5):
        if num_cameras == 1:
            redprint ("Only one camera. You don't need to calibrate.", True)
            return
        
        self.initialize_calibration()
        for _ in range(n_obs)
            process_observation(n_avg)
        self.calibrated = finish_calibration()
        
    def get_transforms(self):
        if self.num_cameras == 1:
            yellowprint("Only have one camera. No transforms.")
            return false
        if not self.calibrated:
            redprint("Cameras not calibrated.")
            return false
        
        return self.camera_transforms
        
        
    def reset_calibration (self):
        self.calibrated = False
        
        for i in streams:
            streams[i]['color'].stop()
            streams[i]['color'].destroy()
            streams[i]['depth'].stop()
            streams[i]['depth'].destroy()
        for i in self.devices:
            self.devices[i].close
        
        self.devices = {}
        self.streams = {}
        self.camera_transforms = []