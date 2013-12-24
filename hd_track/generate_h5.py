#!/usr/bin/env python

"""
Generate hdf5 file based on master files
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("task_file")
parser.add_argument("--cloud_proc_func", default="extract_red")
parser.add_argument("--cloud_proc_mod", default="rapprentice.cloud_proc_funcs")
#parser.add_argument("--no_clouds")
#parser.add_argument("--clouds_only", action="store_true")
args = parser.parse_args()

import os, os.path as osp
import rosbag
import h5py
from rapprentice import bag_proc
import yaml
import importlib, inspect
import numpy as np

task_dir = ops.dirname(args.task_file)

with open(args.task_file, "r") as fh: task_info = yaml.load(fh)
h5path = osp.join(task_dir, task_info["h5path"].strip())

if osp.exists(h5path):
    os.unlink(h5path)
hdf = h5py.File(h5path)

bag_infos = task_info["bags"]

for (i_bag, bag_info) in enumerate(bag_infos):
    bag_file = osp.join(task_dir, bag_info["bag_file"])
    ann_file = osp.join(task_dir, bag_info["annotation_file"])
    video_dirs = bag_info["video_dirs"]
    
    demo_name = bag_info["demo_name"] if "demo_name" in bag_info else "demo%i"%i_bag
    
    bag = rosbag.Bag(bag_file)
    with open(ann_file, "r") as fh: annotations = yaml.load(fh)
    
    

