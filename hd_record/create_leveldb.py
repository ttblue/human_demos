import numpy as np
import leveldb
import argparse
import os, os.path as osp
import cv2
import yaml
import shutil
import caffe
from caffe.io import caffe_pb2
from scipy import linalg
from create_leveldb_utils import *
import cPickle as cp

from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--only_keyframes", action="store_true", help=["Only use keyframes"])
parser.add_argument("--num_samples_per_image", default=20, type=int)
parser.add_argument("--patch_size", default=64, type=int)
parser.add_argument("--test_demo_start", type=int)


args = parser.parse_args()
patch_size = args.patch_size

task_dir = osp.join(demo_files_dir, args.demo_type)
task_file = osp.join(task_dir, master_name)

with open(task_file, "r") as fh: task_info = yaml.load(fh)
ldbpath_train = osp.join(task_dir, "leveldb-train-rand-" + str(patch_size))
ldbpath_test = osp.join(task_dir, "leveldb-test-rand-" + str(patch_size))


if osp.exists(ldbpath_train):
    shutil.rmtree(ldbpath_train)
ldb_train = leveldb.LevelDB(ldbpath_train)

if osp.exists(ldbpath_test):
    shutil.rmtree(ldbpath_test)
ldb_test = leveldb.LevelDB(ldbpath_test)




demos_info = task_info['demos']

if args.only_keyframes == True:
    
    num_train = 0
    num_test = 0
    start_id = 0
    for (index, demo_info) in enumerate(demos_info):
        demo_name = demo_info['demo_name']
        print demo_name
            
        demo_dir = osp.join(task_dir, demo_name)
        rgbd_dir = osp.join(demo_dir, demo_names.video_dir%(1))
        annotation_file = osp.join(demo_dir,"ann.yaml")
        with open(annotation_file, "r") as fh: annotations = yaml.load(fh)
    
        if index < args.test_demo_start:
            start_id = add_rgb_to_leveldb(rgbd_dir, annotations, ldb_train, demo_name, args.patch_size, args.num_samples_per_image, start_id)
            num_train = start_id
        else:
            start_id = add_rgb_to_leveldb(rgbd_dir, annotations, ldb_test, demo_name, args.patch_size, args.num_samples_per_image, start_id)  
else:
    print 'todo'
    
    print 'done'
    
    
