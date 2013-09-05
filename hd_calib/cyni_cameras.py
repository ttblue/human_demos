import cyni
import time
import cv2
import numpy as np

import roslib; roslib.load_manifest('tf')
import rospy, tf

from hd_utils.colorize import *
from hd_utils import conversions

class cyni_cameras:
    """
    This class uses Cyni to calibrate between cameras.
    """

    devices = {}
    streams = {}
    num_cameras = 0
    # Maybe you don't need this if acquiring information is going to take some time.
    emitter_flip_time = 0.5
    flip_emitter = True
    initialized = False
    streaming = {}
    # Transforms calculated by calibration
    camera_transforms = None
    parent_frame = None
    stored_tfms = {}

    def __init__(self, num_cameras=2):
        
        assert num_cameras > 0
        self.num_cameras = num_cameras
        cyni.initialize()
        
        # Corresponds to camera 1
        self.parent_frame = 'camera1_depth_optical_frame'
        
        if rospy.get_name() == '/unnamed':
            rospy.init_node('cam_calibrator')
        self.tf_l = tf.tranformListener()

        
        
    def initialize_cameras(self):
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
            
            self.streaming[i] = False
        
        self.initialized = True
            
    def start_streaming(self, cam=None):
        if cam is not None and self.streaming.get(cam) is False:
            streams[cam]['color'].start()
            streams[cam]['depth'].start()
            streams[cam]['depth'].setEmitterState(False)
            self.streaming[cam] = True
        else:
            for i in streams:
                if self.streaming.get(i) is False:
                    streams[i]['color'].start()
                    streams[i]['depth'].start()
                    streams[i]['depth'].setEmitterState(False)
                    self.streaming[cam] = True

    
    def stop_streaming(self, cam=None):
        if cam is not None and self.streaming[cam] is True:
            streams[cam]['color'].stop()
            streams[cam]['depth'].stop()
            self.streaming[cam] = False
        else:
            for i in streams:
                if self.streaming[cam] is True:
                    streams[i]['color'].stop()
                    streams[i]['depth'].stop()
                    self.streaming[cam] = False

    def get_RGBD (self, cams=None):

        if cams is None or cams[0] is None:
            cams = [i for i in streams]
        data = {i:{} for i in cams}
        for i in cams:
            stream = streams[i]
            if not self.streaming.get(i):
                redprint("Not streaming from camera %d"%i)
                data[i]['depth'] = None
                data[i]['rgb'] = None
                continue
            if flip_emitter: stream["depth"].setEmitterState(True)
            data[i]['depth'] = stream["depth"].readFrame().data
            data[i]['rgb'] = cv2.cvtColor(stream["color"].readFrame().data, cv2.COLOR_RGB2BGR)
            time.sleep(self.emitter_flip_time)
            stream["depth"].setEmitterState(False)
        return data
    
    # Make clique
    def store_calibrated_transforms (self, transforms):
        self.camera_transforms = transforms

    def get_camera_transform(self, c1, c2):
        """
        Gets transform of camera c2 in camera c1's frame.
        """
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
        if frame not in self.stored_tfms:
            trans, rot = self.tf_l.lookupTransform(frame, self.parent_frame, rospy.Time.now(0))
            self.stored_tfms[frame] = conversions.trans_rot_to_hmat(trans, rot)
        
        return self.stored_tfms[frame].dot(self.get_transform(0, cam))