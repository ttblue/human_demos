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
from create_leveldb_utils import *
import cPickle as cp
import h5py

from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name

import caffe
from caffe.proto import caffe_pb2
import scipy



parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--prototxt", help="File name for prototxt")
parser.add_argument("--model", help="File name for learned model")
parser.add_argument("--mean", help="File name for mean values")
args = parser.parse_args()

is_lenet = False
if "lenet" == args.prototxt.split("_")[0]:
    is_lenet = True

if is_lenet:
    net = caffe.Classifier(args.prototxt, args.model)
    net.set_phase_test()
    net.set_mode_gpu()
    net.set_input_scale('data', 1)
    net.set_channel_swap('data', (2,1,0))
else:
    net = caffe.Classifier(args.prototxt, args.model)
    net.set_phase_test()
    net.set_mode_gpu()
    net.set_mean('data', np.load(args.mean))
    net.set_raw_scale('data', 255)
    net.set_channel_swap('data', (2,1,0))

task_dir = osp.join(demo_files_dir, args.demo_type)
train_patches_folder = osp.join(task_dir, "trainData")
test_patches_folder = osp.join(task_dir, "testData")


dict_image_class_rough= {'background': 0, 'cornerA': 1, 'cornerB': 1, 'cornerC': 1, 'edgeA': 2,
                           'edgeB': 2, 'edgeC': 2, 'edgeAB': 2, 'edgeBC': 2, 'edgeAC': 2, 'face': 3}

dict_class_name = {0: 'background', 1: 'corner', 2: 'edge', 3: 'face'}

train_image_classes = [f for f in listdir(train_patches_folder)]
test_image_classes = [f for f in listdir(test_patches_folder)]

n_train_fail = 0
n_train = 0
for image_class in train_image_classes:
    label = dict_image_class_rough[image_class]
    image_dir = osp.join(train_patches_folder, image_class)
    image_names = [f for f in listdir(image_dir)]
    
    for image_name in image_names:
        image = caffe.io.load_image(osp.join(image_dir, image_name))
        scores = net.predict([image])
        score = scores[0]
        predict = score.argmax()
        print image_name, predict, label
        n_train += 1
        if predict != label:
            n_train_fail += 1
            folder_name = osp.join("train_error", dict_class_name[label] + "_to_" + dict_class_name[predict])
            if not osp.exists(folder_name):
                os.makedirs(folder_name)
            print osp.join("train_error", folder_name, image_name)
            scipy.misc.imsave(osp.join(folder_name, image_name), image)

n_test_fail = 0
n_test = 0
for image_class in test_image_classes:
    label = dict_image_class_rough[image_class]
    image_dir = osp.join(test_patches_folder, image_class)
    image_names = [f for f in listdir(image_dir)]
    
    for image_name in image_names:
        image = caffe.io.load_image(osp.join(image_dir, image_name))
        scores = net.predict([image])
        score = scores[0]
        predict = score.argmax()
        print image_name, predict, label
        n_test += 1
        if predict != label:
            n_test_fail += 1
            folder_name = osp.join("test_error", dict_class_name[label] + "_to_" + dict_class_name[predict])
            if not osp.exists(folder_name):
                os.makedirs(folder_name)
            print osp.join("test_error", folder_name, image_name)
            scipy.misc.imsave(osp.join(folder_name, image_name), image)
            
print 1 - n_train_fail / float(n_train)
print 1 - n_test_fail / float(n_test)





