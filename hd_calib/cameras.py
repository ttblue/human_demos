import time
import numpy as np

import rospy
import roslib; roslib.load_manifest('tf')
roslib.load_manifest('ar_track_alvar')
import tf
from ar_track_alvar.msg import AlvarMarkers
from sensor_msgs.msg import PointCloud2

from hd_utils.colorize import *
from hd_utils.defaults import tfm_link_rof
from hd_utils import conversions, utils

class ARMarkersRos:
    """
    Class to store the latest message from ar topic for a given camera.
    """
    latest_markers = None
    latest_time = 0.0
    freq = 0.0
    count = 0
    alpha = 0.8
    
    def __init__(self, camera_frame):
        
        if camera_frame[0] == "/":
            camera_frame = camera_frame[1:]
        
        self.camera_name = camera_frame.split("_")[0]
        self.marker_topic = 'ar_pose_marker_' + self.camera_name
        
        if rospy.get_name() == '/unnamed':
            rospy.init_node('ar_markers_' + self.camera_name)
        
        self.ar_sub = rospy.Subscriber(self.marker_topic, AlvarMarkers, callback=self.store_last_callback) 
            
    def store_last_callback (self, data):
        if len(data.markers) == 0:
            if self.freq != 0.0:
                time_now = data.header.stamp.to_sec()
                if time_now == 0.0: time_now = rospy.Time.now().to_sec()
                if time_now - self.latest_time > 3.0 / self.freq:
                    self.freq = 0.0
            return
        
        # estimate frequency from sequential data
        self.count = (self.count + 1) % 100000
        self.latest_markers = data
        
        time_now = data.header.stamp.to_sec()
        if time_now == 0.0: time_now = rospy.Time.now().to_sec()
        if self.latest_time != 0.0:
            if self.freq == 0.0:
                self.freq = 1.0 / (time_now - self.latest_time)
            else:
                self.freq = (1.0 - self.alpha) * self.freq + self.alpha / (time_now - self.latest_time)
        self.latest_time = time_now      
    
    def get_frequency (self):
        return self.freq 
        
    def get_marker_transforms(self, markers=None, time_thresh=1.0, get_time=False):
        """
        Get the transforms for markers denoted by their ids (markers)
        Threshold represents the tolerance for stale transforms.
        """
        time_now = rospy.Time.now().to_sec()
        if self.latest_markers is None or time_now - self.latest_time > time_thresh: 
            if get_time:
                return {}, self.latest_time
            else:
                return {}

        if markers is None:
            marker_transforms = {marker.id:conversions.pose_to_hmat(marker.pose.pose)\
                                 for marker in self.latest_markers.markers if marker.id != 0}
        else:
            marker_transforms = {marker.id:conversions.pose_to_hmat(marker.pose.pose)\
                                 for marker in self.latest_markers.markers if marker.id in markers and marker.id != 0}

        if not get_time:
            return marker_transforms
        else:
            return marker_transforms, self.latest_time


class CameraData:

    def __init__(self, camera_frame, collect_pc=True, collect_image=False):
        if camera_frame[0] == "/":
                camera_frame = camera_frame[1:]
            
        self.camera_name = camera_frame.split("_")[0]

        self.latest_time_pc = 0
        self.latest_time_img = 0
        
        if rospy.get_name() == '/unnamed':
            rospy.init_node('point_clouds_' + self.camera_name)
            
        if collect_pc:
            self.pc_topic = '/' + self.camera_name + '/depth_registered/points'
            self.pc = None
            self.pc_sub = rospy.Subscriber(self.pc_topic, PointCloud2, callback=self.store_last_callback_pc)

        
        if collect_image:
            self.image_topic = '/' + self.camera_name + '/rgb/image_color'
            self.image = None
            self.image_sub = rospy.Subscriber(self.image_topic, Image, callback = self.store_last_callback_image)
        
    def store_last_callback_pc (self, data):
        self.pc = data
        self.latest_time_pc = data.header.stamp.to_sec()

    def store_last_callback_image (self, data):
        self.image = data
        self.latest_time_img = data.header.stamp.to_sec()
        
    def get_latest_pointcloud(self):
        if self.pc is None:
            redprint("No point clouds has been received yet on topic", self.pc_topic)
        else:
            return self.pc

    def get_latest_image(self):
        if self.image is None:
            redprint("No image has been received yet on topic", self.image_topic)
        else:
            return self.image

class RosCameras:
    """
    This class uses ROS to get camera data.
    """
    # Making assumptions on the frames.
    camera_frames = {}
    camera_markers = {}
    camera_pointclouds = {}
        
    calibrated = False
    
    camera_transforms = None
    parent_frame = 'camera1_rgb_optical_frame'
    stored_tfms = {}

    def __init__(self, num_cameras=2):
        assert num_cameras > 0
        # Corresponds to camera 1

        self.num_cameras = num_cameras
        self.calibrated = (self.num_cameras == 1)
        
        camera_frame = 'camera%d_rgb_optical_frame'
        
        if rospy.get_name() == '/unnamed':
            rospy.init_node('cam_calibrator')
        
        for i in xrange(self.num_cameras):
            camera_frame_name = camera_frame % (i + 1)
            self.camera_frames[i] = camera_frame_name
            self.camera_markers[i] = ARMarkersRos(camera_frame_name)
            self.camera_pointclouds[i] = CameraData(camera_frame_name) 
    
    def get_ar_markers (self, markers=None, camera=None, parent_frame=False, get_time=False):
        """
        @markers is a list of markers to be found. Default of None means all markers.
        @camera specifies which camera to use. Default of None means all cameras are used 
            (assuming calibrated).
        @parent_frame specifies whether the transforms are in the parent frame or camera frame
            (assuming calibrated).
        
        
        UGLY FUNCTION -- BREAK IT INTO BETTER FUNCTIONS.
        """
        if camera is None:
            if not self.calibrated:
                redprint('Cameras not calibrated. Cannot get transforms from all cameras.')
                return {}
            
            marker_tfms = {}
            time_stamp = 0.0
            num_seen = 0
            for i in range(self.num_cameras):
                ctfm = self.get_camera_transform(0, i)
                tfms, t = self.camera_markers[i].get_marker_transforms(markers, get_time=True)
                if tfms: 
                    time_stamp += t
                    num_seen += 1
                
                
                for marker in tfms:
                    if marker not in marker_tfms:
                        marker_tfms[marker] = []
                    marker_tfms[marker].append(ctfm.dot(tfms[marker]))
                
            for marker in marker_tfms:
                marker_tfms[marker] = utils.avg_transform(marker_tfms[marker])
            if num_seen > 0:
                time_stamp = time_stamp / num_seen
        else:
            assert camera in range(self.num_cameras)
            marker_tfms, time_stamp = self.camera_markers[camera].get_marker_transforms(markers, get_time=True)
            if parent_frame is True:
                if not self.calibrated:
                    redprint('Cameras not calibrated. Cannot get transforms from all cameras.')
                    return {}
                ctfm = self.get_camera_transform(0, camera)
                for marker, tfm in marker_tfms.items():
                    marker_tfms[marker] = ctfm.dot(tfm) # all transforms in reference camera's frame

        if get_time:
            return marker_tfms, time_stamp
        else:
            return marker_tfms
        
    def get_pointcloud(self, camera=0):
        if camera not in range(self.num_cameras):
            redprint("Camera %i out of range." % camera)
        return self.camera_pointclouds[camera].get_latest_pointcloud()
    
    def get_checkerboard_points (self, rows, cols):
        pass
    

    def store_calibrated_transforms (self, transforms):
        assert self.calibrated
        self.camera_transforms = transforms

    def get_camera_transform(self, c1, c2):
        """
        Gets transform of camera c2 in camera c1's frame. 
        (In fact transform of camera c2's camera_rgb_optical_frame in camera c1's camera_rgb_optical_frame)
        TODO: Make clique to get any transform.
        """
        if self.calibrated is False:
            raise Exception('Cameras not calibrated.')
        if self.num_cameras == 1:
            return np.eye(4)
        if c1 == c2:
            return np.eye(4)
        if self.camera_transforms.get((c1, c2)) is None and self.camera_transforms.get((c2, c1)) is None:
            print "Transform not found."
            return None
        tfm = self.camera_transforms.get((c1, c2))
        if tfm is not None:
            return np.linalg.inv(tfm_link_rof).dot(tfm['tfm']).dot(tfm_link_rof)
        else:
            tfm = self.camera_transforms.get((c2, c1))
            return np.linalg.inv(tfm_link_rof).dot(np.linalg.inv(tfm['tfm'])).dot(tfm_link_rof)
