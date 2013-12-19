#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from threading import Thread
import time
import os
import os.path as osp
import cPickle

import roslib
import rospy
roslib.load_manifest('tf')
import tf
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32

from hd_utils.colorize import *
from hd_utils.yes_or_no import yes_or_no
from hd_utils import conversions, clouds, ros_utils as ru
from hd_utils.defaults import asus_xtion_pro_f, calib_files_dir

from cameras import RosCameras
from camera_calibration import CameraCalibrator
from hydra_calibration import HydraCalibrator
from gripper_calibration import GripperCalibrator
import gripper
import gripper_lite
import get_marker_transforms as gmt

import read_arduino

finished = False


def done():
    global finished
    finished = True


np.set_printoptions(precision=5, suppress=True)

gripper_trans_marker_tooltip = {'l': [0.10398, -0.00756, -0.03999],
                                'r': [0.07783, 0.0, -0.04416]}
gripper_marker_id = {'l': 1, 'r': 4}


class CalibratedTransformPublisher(Thread):

    def __init__(self, cameras=None):
        Thread.__init__(self)

        if rospy.get_name() == '/unnamed':
            rospy.init_node('calibration_node')

        self.transforms = {}
        self.ready = False
        self.rate = 30.0
        self.tf_broadcaster = tf.TransformBroadcaster()

        self.langle_pub = rospy.Publisher('l_pot_angle', Float32)
        self.rangle_pub = rospy.Publisher('r_pot_angle', Float32)

        self.cameras = cameras
        self.publish_grippers = False
        self.gripper_lite = True
        self.grippers = {}

    def run(self):
        """
        Publishes the transforms stored.
        """
        while True and not finished:
            if self.ready:
                for parent, child in self.transforms:
                    trans, rot = self.transforms[parent, child]
                    self.tf_broadcaster.sendTransform(trans, rot,
                            rospy.Time.now(), child, parent)
                if self.publish_grippers:
                    self.publish_gripper_tfms()
            time.sleep(1 / self.rate)

    def add_transforms(self, transforms):
        """
        Takes a list of dicts with relevant transform information.
        """

        if transforms is None:
            return
        for transform in transforms:
            (trans, rot) = conversions.hmat_to_trans_rot(transform['tfm'])
            self.transforms[transform['parent'], transform['child']] = (trans, rot)
        self.ready = True
        
    def remove_transform(self, parent, child):
        if (parent, child) in self.transforms:
            self.transforms.pop((parent, child))

    def get_all_transforms(self):
        '''
        Return [{'parent': parent, 'child': child, 'tfm': hmat}]
        '''

        result_tfms = []
        for (parent, child) in self.transforms:
            tfm = {'parent': parent, 'child': child}
            (trans, rot) = self.transforms[parent, child]
            tfm['tfm'] = conversions.trans_rot_to_hmat(trans, rot)
            result_tfms.append(tfm)

        return result_tfms

    def get_camera_transforms(self):
        '''
        Return a dictionary {(c1, c2): tf1, (c1', c2'): tf2}
        '''

        result_tfms = {}
        for (parent, child) in self.transforms:
            if 'camera' in parent and 'camera' in child:
                c1 = int(parent[6]) - 1  # assume parent is something like 'camera1'
                c2 = int(child[6]) - 1  # assume child is something like 'camera1'
                tfm = {'parent': parent, 'child': child}
                (trans, rot) = self.transforms[parent, child]
                tfm['tfm'] = conversions.trans_rot_to_hmat(trans, rot)
                result_tfms[c1, c2] = tfm
        return result_tfms

    def add_gripper(self, gr):
        self.grippers[gr.lr] = gr

    def publish_gripper_tfms(self):
        marker_tfms = self.cameras.get_ar_markers()

        self.langle_pub.publish(gmt.get_pot_angle('l'))
        self.rangle_pub.publish(gmt.get_pot_angle('r'))

        transforms = []
        for gr in self.grippers.values():
            if self.gripper_lite:
                transforms += gr.get_all_transforms(diff_cam=True)
            else:
                parent_frame = self.cameras.parent_frame
                transforms += gr.get_all_transforms(parent_frame, diff_cam=True)

        for transform in transforms:
            (trans, rot) = conversions.hmat_to_trans_rot(transform['tfm'])
            self.tf_broadcaster.sendTransform(trans, rot,
                                              rospy.Time.now(), transform['child'],
                                              transform['parent'])

    def set_publish_grippers(self, val=None):
        """
        Toggles by default
        """

        if val is None:
            self.publish_grippers = not self.publish_grippers
        else:
            self.publish_grippers = not not val

        if self.publish_grippers:
            self.ready = True

    def reset(self, grippers_only=False):
        if not grippers_only:
            self.ready = False
            self.transforms = {}

        self.publish_grippers = False
        self.grippers = {}

    def load_calibration(self, file):
        """
        Use this if experimental setup has not changed.
        Load files which have been saved by this class. Specific format involved.
        """

        self.reset()

        file_name = osp.join(calib_files_dir, file)
        with open(file_name, 'r') as fh:
            calib_data = cPickle.load(fh)
        if self.gripper_lite:
            for (lr, data) in calib_data['grippers'].items():
                gr = gripper_lite.GripperLite(lr, data['ar'],
                                              trans_marker_tooltip=gripper_trans_marker_tooltip[lr],
                                              cameras=self.cameras)
                gr.reset_gripper(lr, data['tfms'], data['ar'],
                                 data['hydra'])
                self.add_gripper(gr)
        else:
            for (lr, graph) in calib_data['grippers'].items():
                gr = gripper.Gripper(lr, graph, self.cameras)
                if 'tool_tip' in gr.mmarkers:
                    gr.tooltip_calculated = True
                self.add_gripper(gr)
        self.add_transforms(calib_data['transforms'])

        if self.grippers:
            self.publish_grippers = True
        if self.cameras is not None:
            self.cameras.calibrated = True
            self.cameras.store_calibrated_transforms(self.get_camera_transforms())

    def load_gripper_calibration(self, file):
        """
        Use this if gripper markers have not changed.
        Load files which have been saved by this class. Specific format involved.
        """

        self.reset(grippers_only=True)

        file_name = osp.join(calib_files_dir, file)
        with open(file_name, 'r') as fh:
            calib_data = cPickle.load(fh)

        if self.gripper_lite:
            for (lr, data) in calib_data['grippers'].items():
                print lr
                print data
                gr = gripper_lite.GripperLite(lr, data['ar'],
                                              trans_marker_tooltip=gripper_trans_marker_tooltip[lr],
                                              cameras=self.cameras)
                gr.reset_gripper(lr, data['tfms'], data['ar'],
                                 data['hydra'])
                self.add_gripper(gr)
        else:
            for (lr, graph) in calib_data['grippers'].items():
                gr = gripper.Gripper(lr, graph, self.cameras)
                if 'tool_tip' in gr.mmarkers:
                    gr.tooltip_calculated = True
                self.add_gripper(gr)

        self.publish_grippers = True

    def save_calibration(self, file):
        """
        Save the transforms and the gripper data from this current calibration.
        This assumes that the entire calibration data is stored in this class.
        """

        calib_data = {}

        gripper_data = {}
        if self.gripper_lite:
            for (lr, gr) in self.grippers.items():
                gripper_data[lr] = {'ar': gr.get_ar_marker(),
                                    'hydra': gr.get_hydra_marker(),
                                    'tfms': gr.get_saveable_transforms()}
        else:
            for (lr, gr) in self.grippers.items():
                gripper_data[lr] = gr.transform_graph

        calib_data['grippers'] = gripper_data

        calib_transforms = []
        for (parent, child) in self.transforms:
            tfm = {}
            tfm['parent'] = parent
            tfm['child'] = child
            (trans, rot) = self.transforms[parent, child]
            tfm['tfm'] = conversions.trans_rot_to_hmat(trans, rot)
            calib_transforms.append(tfm)
        calib_data['transforms'] = calib_transforms

        file_name = osp.join(calib_files_dir, file)
        with open(file_name, 'w') as fh:
            cPickle.dump(calib_data, fh)


def calibrate_potentiometer(lr='l'):
    if not gmt.pot_initialized:
        gmt.arduino = read_arduino.Arduino()
        pot_initialized = True
        print 'POT INITIALIZED'

    yellowprint('Calibrating potentiometer:')
    raw_input(colorize('Close gripper all the way to 0 degrees.',
              'yellow', True))
    gmt.b[lr] = gmt.arduino.get_reading()

    raw_input(colorize('Now, open gripper all the way.', 'yellow',
              True))
    gmt.a[lr] = (gmt.arduino.get_reading() - gmt.b[lr]) / 30.0

    greenprint('Potentiometer calibrated!')


# Global variables

cameras = None
tfm_pub = None

CAM_N_OBS = 2
CAM_N_AVG = 30


def calibrate_cameras():
    global cameras, tfm_pub

    greenprint('Step 1. Calibrating multiple cameras.')
    cam_calib = CameraCalibrator(cameras)

    done = False
    while not done:
        cam_calib.calibrate(n_obs=CAM_N_OBS, n_avg=CAM_N_AVG)
        if cameras.num_cameras == 1:
            break
        if not cam_calib.calibrated:
            redprint('Camera calibration failed.')
            cam_calib.reset_calibration()
        else:
            tfm_pub.add_transforms(cam_calib.get_transforms())
            if yes_or_no('Are you happy with the calibration? Check RVIZ.'
                         ):
                done = True
            else:
                yellowprint('Calibrating cameras again.')
                cam_calib.reset_calibration()

    greenprint('Cameras calibrated.')


HYDRA_N_OBS = 15
HYDRA_N_AVG = 50
CALIB_CAMERA = 0


def calibrate_hydras():
    global cameras, tfm_pub

    greenprint('Step 2. Calibrating hydra and kinect.')
    HYDRA_AR_MARKER = \
        input('Enter the ar_marker you are using for calibration.')
    hydra_calib = HydraCalibrator(cameras, ar_marker = HYDRA_AR_MARKER,
                                  calib_hydra = 'left',
                                  calib_camera = CALIB_CAMERA)

    done = False
    while not done:
        hydra_calib.calibrate('camera', HYDRA_N_OBS, HYDRA_N_AVG)
        if not hydra_calib.calibrated:
            redprint('Hydra calibration failed.')
            hydra_calib.reset_calibration()
        else:
            tfm_pub.add_transforms(hydra_calib.get_transforms('camera'))
            if yes_or_no('Are you happy with the calibration? Check RVIZ.'):
                done = True
            else:
                yellowprint('Calibrating hydras.')
                hydra_calib.reset_calibration()

    greenprint('Hydra base calibrated.')


GRIPPER_MIN_OBS = 4

# markers on gripper fingers (not used for gripper_lite)
gripper_finger_markers = {'l': {'l': 15, 'r': 4}, # left gripper
                          'r': {'l': 16, 'r': 5}} # right gripper

def calibrate_grippers():
    greenprint('Step 3. Calibrating relative transforms of markers on gripper.')

    calibrate_gripper('l')
    calibrate_gripper('r')


def calibrate_gripper(lr):
    global cameras, tfm_pub


    if lr == 'l':
        greenprint('Step 3.1 Calibrate l gripper.')
    else:
        greenprint('Step 3.2 Calibrate r gripper')
    
    gripper_calib = GripperCalibrator(cameras, lr)

    # create calib_info based on gripper here

    calib_info = {'master': {'ar_markers': [gripper_marker_id[lr]],
                             'angle_scale': 0,
                             'master_marker': gripper_marker_id[lr]},
                    'l': {'ar_markers': [gripper_finger_markers[lr]['l']],
                          'angle_scale': 1},
                    'r': {'ar_markers': [gripper_finger_markers[lr]['r']],
                          'angle_scale': -1}}

    gripper_calib.update_calib_info(calib_info)

    done = False
    while not done:
        gripper_calib.calibrate(GRIPPER_MIN_OBS, GRIPPER_N_AVG)
        if not gripper_calib.calibrated:
            redprint('Gripper calibration failed.')
            gripper_calib.reset_calibration()

        tfm_pub.add_gripper(gripper_calib.get_gripper())
        tfm_pub.set_publish_grippers(True)

        if yes_or_no('Are you happy with the calibration?'):
            done = True
        else:
            yellowprint('Calibrating ' + lr + ' gripper again.')
            gripper_calib.reset_calibration()
            gripper_calib.update_calib_info(calib_info)
            tfm_pub.set_publish_grippers(False)

    greenprint('Done with ' + lr + ' gripper calibration.')
    

def calibrate_grippers_lite():
    greenprint('Step 3. Calibrating relative transforms of markers on gripper.')
    
    calibrate_gripper_lite('l')
    calibrate_gripper_lite('r')
    
def calibrate_gripper_lite(lr):
    global cameras, tfm_pub

    if lr == 'l':
        greenprint('Step 3.1 Calibrate l gripper.')
    else:
        greenprint('Step 3.1 Calibrate r gripper.')
        
    gr = gripper_lite.GripperLite(lr, marker=gripper_marker_id[lr], 
                                  cameras=cameras,
                                  trans_marker_tooltip=gripper_trans_marker_tooltip[lr])     
        
    tfm_pub.add_gripper(gr)
    tfm_pub.set_publish_grippers(True)

    if lr == 'l':
        if yes_or_no(colorize('Do you want to add hydra to lgripper?', 'yellow')):
            gr.add_hydra(hydra_marker='left', tfm=None, ntfm=50, navg=30)
    else:
        if yes_or_no(colorize('Do you want to add hydra to rgripper?', 'yellow')):
            gr.add_hydra(hydra_marker='right', tfm=None, ntfm=50, navg=30)
            
    

NUM_CAMERAS = 2

def initialize_calibration(num_cams=NUM_CAMERAS):
    global cameras, tfm_pub
    rospy.init_node('calibration', anonymous=True)
    cameras = RosCameras(num_cameras=num_cams)
    tfm_pub = CalibratedTransformPublisher(cameras)

    # tfm_pub.fake_initialize()

    tfm_pub.start()


def run_calibration_sequence(spin=False):

    yellowprint('Beginning calibration sequence.')
    initialize_calibration()
    calibrate_cameras()
    calibrate_hydras()
    calibrate_grippers()

    greenprint('Done with all the calibration.')

    while True and spin:

        # stall

        time.sleep(0.1)


# if __name__=='__main__':
#     initialize_calibration(1)
#     tfm_pub.load_calibration('cc')
#     tfm_pub.load_calibration('cam12HG_calib')
#
