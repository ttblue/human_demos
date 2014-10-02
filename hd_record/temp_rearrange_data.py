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
import h5py
from os import listdir


from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--in_folder", type=str)
args = parser.parse_args()

task_dir = osp.join(demo_files_dir, args.demo_type)
in_folder = osp.join(task_dir, args.in_folder)
out_folder = osp.join(task_dir, args.in_folder+"2")

os.makedirs(out_folder)

image_names = [f for f in listdir(in_folder)]
for image_name in image_names:
    words = image_name.split(".")
    image_id = int(words[0])
    if image_id % 26 == 0:
        shutil.move(osp.join(in_folder, image_name), osp.join(out_folder, image_name))
        

        
