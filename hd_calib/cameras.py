import time
import numpy as np

import rospy
import roslib; roslib.load_manifest('tf')
roslib.load_manifest('ar_track_alvar')
import tf
from ar_track_alvar.msg import AlvarMarkers

from hd_utils.colorize import *
from hd_utils import conversions, utils

class ar_markers_ros:
    """
    Class to store the latest message from ar topic for a given camera.
    """
    latest_markers = None
    latest_time = 0
    
    def __init__(self, camera_frame):
        
        if camera_frame[0] == "/":
            camera_frame = camera_frame[1:]
        
        self.camera_name = camera_frame.split("_")[0]
        self.marker_topic = 'ar_pose_marker_'+self.camera_name
        
        if rospy.get_name() == '/unnamed':
            rospy.init_node('ar_markers_'+self.camera_name)
        
        self.ar_sub = rospy.Subscriber(self.marker_topic, AlvarMarkers, callback = self.store_last_callback) 
            
    def store_last_callback (self, data):
        if len(data.markers) == 0:
            return
        self.latest_markers = data
        self.latest_time = data.header.stamp.to_sec()
        
    def get_marker_transforms(self, markers=None, time_thresh=1):
        """
        Threshold represents the tolerance for stale transforms.
        """
        if self.latest_markers is None: return {}
        
        time_now = rospy.Time.now().to_sec()
        if time_now - self.latest_time > time_thresh: return {}
        
        if markers is None:
            marker_transforms = {marker.id:conversions.pose_to_hmat(marker.pose.pose)\
                                 for marker in self.latest_markers.markers}
        else:
            marker_transforms = {marker.id:conversions.pose_to_hmat(marker.pose.pose)\
                                 for marker in self.latest_markers.markers if marker.id in markers}

        return marker_transforms

class ros_cameras:
    """
    This class uses ROS to get camera data.
    """
    # Making assumptions on the frames.
    camera_frames = {}
    camera_markers = {}
    
    camera_transforms = None
    parent_frame = None
    stored_tfms = {}

    def __init__(self, num_cameras=2):
        assert num_cameras > 0
        # Corresponds to camera 1

        self.num_cameras = num_cameras
        
        camera_frame = 'camera%d_depth_optical_frame'
        
        if rospy.get_name() == '/unnamed':
            rospy.init_node('cam_calibrator')
        
        for i in xrange(self.num_cameras):
            self.camera_frames[i] = camera_frame%(i+1)
            self.camera_markers[i] = ar_markers_ros(camera_frame%(i+1))
                
        self.parent_frame = 'camera1_depth_optical_frame'
    
    def get_ar_markers (self, markers=None, camera=None, parent_frame=False):
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
            
            markers_tfms = {}
            for i in range(self.num_cameras):
                ctfm = self.get_camera_transform(0, i)
                tfms = self.camera_markers[i].get_marker_transforms(markers)
                
                for marker in tfms:
                    if marker not in marker_tfms:
                        marker_tfms[marker] = []
                    marker_tfms[marker].append(ctfm.dot(tfms[marker]))
                
            for marker in marker_tfms:
                marker_tfms[marker] = utils.avg_transform(marker_tfms[marker])
        else:
            assert camera in range(self.num_cameras)
            marker_tfms = self.camera_markers[camera].get_marker_transforms(markers)
            if parent_frame is True:
                if not self.calibrated:
                    redprint('Cameras not calibrated. Cannot get transforms from all cameras.')
                    return {}
                ctfm = self.get_camera_transform(0, camera)
                for marker,tfm in marker_tfms.items():
                    marker_tfms[marker] = ctfm.dot(tfm)

            
        return marker_tfms
    
    # Make clique
    def store_calibrated_transforms (self, transforms):
        assert self.calibrated
        self.camera_transforms = transforms

    def get_camera_transform(self, c1, c2):
        """
        Gets transform of camera c2 in camera c1's frame.
        TODO: Make clique to get any transform.
        """
        if self.calibrated is False:
            raise Exception('Cameras not calibrated.')

        if c1==c2:
            return np.eye(4)
        elif self.camera_transforms.get((c1,c2)) is None or self.camera_transforms.get((c2,c1)) is None:
            print "Transform not found."
            return None
        tfm = self.camera_transforms.get((c1,c2))
        if tfm is not None:
            return tfm['tfm']
        else:
            tfm = self.camera_transforms.get((c2,c1))
            return np.linalg.inv(tfm['tfm'])

    
    def get_transform_frame(self, cam, frame):
        """
        Gets transform of cam in frame's frame.
        Stores transforms when found. Assumes offset does not change between frame and camera.
        """
        if frame in self.camera_frames.values() and self.calibrated:
            cam0 = [i for i in self.camera_frames if self.camera_frames[i]==frame][0]
            return self.get_camera_transform(cam0, cam)
        else:
            redprint('Camera not calibrated. Could not get transform.')
            return None

        if frame not in self.stored_tfms:
            trans, rot = self.tf_l.lookupTransform(frame, self.parent_frame, rospy.Time(0))
            self.stored_tfms[frame] = conversions.trans_rot_to_hmat(trans, rot)
        
        return self.stored_tfms[frame].dot(self.get_camera_transform(0, cam))