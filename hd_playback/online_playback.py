'''
Script to playback data on the PR2 in real-time during recording
'''

import argparse
import os.path as osp
import yaml
import time
import rospy
import numpy as np, numpy.linalg as nlg
import cPickle as pickle
import IPython
import cv2, cv

from hd_utils.defaults import demo_files_dir, demo_names, master_name, \
    calib_files_dir, tfm_link_rof, asus_xtion_pro_f
from hd_utils.colorize import *
from hd_utils.yes_or_no import yes_or_no


from hd_utils import clouds, ros_utils as ru, conversions

from hd_calib import gripper_lite, cameras
from hd_calib.calibration_pipeline import gripper_trans_marker_tooltip


def load_calib(num_cameras, calib_file):

    c_frames = {}
    for i in range(1, num_cameras + 1):
        c_frames[i]= 'camera%i_link'%(i)
    hydra_frame = 'hydra_base'

    tfm_c1 = {i:None for i in range (1,num_cameras+1)}
    tfm_c1[1] = np.eye(4)
    tfm_c1_h = None

    calib_file_path = osp.join(calib_files_dir, calib_file)
    with open(calib_file_path,'r') as fh: calib_data = pickle.load(fh)

    for tfm in calib_data['transforms']:
        if tfm['parent'] == c_frames[1] or tfm['parent'] == '/' + c_frames[1]:
            if tfm['child'] == hydra_frame or tfm['child'] == '/' + hydra_frame:
                tfm_c1_h = nlg.inv(tfm_link_rof).dot(tfm['tfm'])
            else:
                for i in range(2, num_cameras+1):
                    if tfm['child'] == c_frames[i] or tfm['child'] == '/' + c_frames[i]:
                        tfm_c1[i] = nlg.inv(tfm_link_rof).dot(tfm['tfm']).dot(tfm_link_rof)


    grippers = {}
    for lr,gdata in calib_data['grippers'].items():
        gr = gripper_lite.GripperLite(lr, gdata['ar'], trans_marker_tooltip=gripper_trans_marker_tooltip[lr])
        gr.reset_gripper(lr, gdata['tfms'], gdata['ar'], gdata['hydra'])
        grippers[lr] = gr

    return grippers, tfm_c1, tfm_c1_h


# For getting tt tfms from cameras
def latest_camera_tt_tfm(num_cameras, grippers, tfm_c1, tfm_c1_h):
    camera_tt_tfm = {'l': {'camera1': None, 'camera2': None, 'camera3': None}, 'r': {'camera1': None, 'camera2': None, 'camera3': None}}
    camera_frame = 'camera%d_rgb_optical_frame'
    for i in range(1,num_cameras+1):
        camera_frame_name = camera_frame % (i + 1)
        self.camera_markers[i] = ARMarkersRos(camera_frame_name)
        ar_tfms= self.camera_markers[i].get_marker_transforms()
        for lr,gr in grippers.items():
            if ar in ar_tfms:
                tt_tfm = gr.get_tooltip_transform(ar, np.asarray(ar_tfms[ar]))
                camera_tt_tfm[lr]['camera%i'%i] = tfm_c1[i].dot(tt_tfm)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cameras", help="number of cameras", action='store', dest='num_cameras', type=int, default=3)
    parser.add_argument("--calib_file", help="calib file to use", action='store', dest='calib_file', type=str)
    args = parser.parse_args()

    grippers, tfm_c1, tfm_c1_h = load_calib(args.num_cameras, args.calib_file)
    start = time.time()
    for i in xrange(100):
        print i
        latest_camera_tt_tfm(args.num_cameras, grippers, tfm_c1, tfm_c1_h)
    print time.time() - start

