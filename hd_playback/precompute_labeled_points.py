import numpy as np
import leveldb
import argparse
import os, os.path as osp
from os import listdir
import cv2
import yaml
import shutil
import caffe
from caffe.io import caffe_pb2
from scipy import linalg
import cPickle as cp
import h5py

from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name

from knot_predictor import predictCrossing2D

import caffe
from caffe.proto import caffe_pb2
import scipy


parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--h5_name", help="Name of h5 file", type=str, default="")
parser.add_argument("--net_prototxt", help="File name for prototxt")
parser.add_argument("--net_model", help="File name for learned model")
parser.add_argument("--net_mean", help="File name for mean values")
args = parser.parse_args()



is_lenet = False
if "lenet" == args.net_prototxt.split("_")[0]:
    is_lenet = True

if is_lenet:
    net = caffe.Classifier(args.net_prototxt, args.net_model)
    net.set_phase_test()
    net.set_mode_gpu()
    net.set_input_scale('data', 1)
    net.set_channel_swap('data', (2,1,0))
else:
    net = caffe.Classifier(args.net_prototxt, args.net_model)
    net.set_phase_test()
    net.set_mode_gpu()
    net.set_mean('data', np.load(args.net_mean))
    net.set_raw_scale('data', 255)
    net.set_channel_swap('data', (2,1,0))

    
    
params = [v.data.shape for k, v in net.blobs.items()]
n_parallel = params[0][0]
patch_size = params[0][2]
offset = np.round(patch_size / 2.).astype(int)

    
task_dir = osp.join(demo_files_dir, args.demo_type)

if args.h5_name == "":
    h5_name = args.demo_type + ".h5"
else:
    h5_name = args.h5_name + ".h5"
    
demos = h5py.File(osp.join(task_dir, h5_name))

for demo_name in demos:
    for seg_name in demos[demo_name]:
        seg_group = demos[demo_name][seg_name]
        
        if 'labeled_points' in seg_group.keys():
            labeled_points = seg_group["labeled_points"]
            image = seg_group["rgb"]
            xy = []
            for i in range(len(labeled_points)):
                (x,y,c) = labeled_points[i, :]
                xy.append((x,y))
            
            xy = np.asarray(xy)
            
            import IPython
            IPython.embed()
            rope_crossing_predicts = predictCrossing2D(xy, image, net)
            
            learned_labeled_points = []
            for i in range(len(labeled_points)):
                x, y, c = labeled_points[i, 0], labeled_points[i, 1], rope_crossing_predicts[i]
                learned_labeled_points.append((x,y,c))
            learned_labeled_points = np.asarray(learned_labeled_points)
            
            seg_group["learned_labeled_points"] = learned_labeled_points

                
            
                
        
    
    
    
    

  


