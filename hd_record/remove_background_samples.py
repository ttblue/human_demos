import numpy as np
import leveldb
import argparse
import os, os.path as osp
from os import listdir
import cv2
import yaml
import shutil
import caffe
from create_leveldb_utils import *

from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name
from hd_utils.extraction_utils import get_video_frames
import cPickle as cp

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
args = parser.parse_args()

task_dir = osp.join(demo_files_dir, args.demo_type)
train_patches_folder = osp.join(task_dir, "trainData_raw")
test_patches_folder= osp.join(task_dir, "testData_raw")

train_bg_folder = osp.join(task_dir, "trainData_bg")
test_bg_folder= osp.join(task_dir, "testData_bg")


train_image_names = [f for f in listdir(train_patches_folder)]
test_image_names = [f for f in listdir(test_patches_folder)]

test_ref_image_name = 'rgb00268.jpg'
train_ref_image_name = 'rgb00030.jpg'

train_rgb_ref = cv2.imread(osp.join(train_patches_folder, train_ref_image_name))
test_rgb_ref = cv2.imread(osp.join(test_patches_folder, test_ref_image_name))

for image_filename in train_image_names:
    rgb = cv2.imread(osp.join(train_patches_folder, image_filename))
    rgb_diff = np.abs(rgb - train_rgb_ref)
    [h,w,c] = rgb_diff.shape
    rgb_diff = rgb_diff.reshape(h*w, c)
    e = [np.linalg.norm(v) for v in rgb_diff]
    #print np.max(e), image_filename
    if np.max(e) < 25:
        print image_filename
        shutil.move(osp.join(train_patches_folder, image_filename), train_bg_folder)
    
    
for image_filename in test_image_names:
    rgb = cv2.imread(osp.join(test_patches_folder, image_filename))
    rgb_diff = np.abs(rgb - test_rgb_ref)
    [h,w,c] = rgb_diff.shape
    rgb_diff = rgb_diff.reshape(h*w, c)
    e = [np.linalg.norm(v) for v in rgb_diff]
    #print np.max(e), image_filename
    if np.max(e) < 25:
        print image_filename
        shutil.move(osp.join(test_patches_folder, image_filename), test_bg_folder)
    
