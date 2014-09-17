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
import cPickle as cp
import h5py
from os import listdir

from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name
import create_leveldb_utils

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--patch_size", default=64, type=int)
parser.add_argument("--test_demo_start", type=int)


args = parser.parse_args()

patch_size = args.patch_size

        
task_dir = osp.join(demo_files_dir, args.demo_type)
h5_filename = osp.join(task_dir, args.demo_type + "_crossing.h5")

ldbpath_train = osp.join(task_dir, "leveldb-train-crossing-" + str(patch_size))
ldbpath_test = osp.join(task_dir, "leveldb-test-crossing-" + str(patch_size))

rawimage_path_train = osp.join(task_dir, "train-crossing-" + str(patch_size))
rawimage_path_test = osp.join(task_dir, "test-crossing-" + str(patch_size))


labeled_train_class_0_names = [f for f in listdir(osp.join(task_dir, "train-crossing", "0"))]
labeled_train_class_1_names = [f for f in listdir(osp.join(task_dir, "train-crossing", "1"))]
labeled_train_class_2_names = [f for f in listdir(osp.join(task_dir, "train-crossing", "2"))]
labeled_train_class_3_names = [f for f in listdir(osp.join(task_dir, "train-crossing", "3"))]

labeled_test_class_0_names = [f for f in listdir(osp.join(task_dir, "test-crossing", "0"))]
labeled_test_class_1_names = [f for f in listdir(osp.join(task_dir, "test-crossing", "1"))]
labeled_test_class_2_names = [f for f in listdir(osp.join(task_dir, "test-crossing", "2"))]
labeled_test_class_3_names = [f for f in listdir(osp.join(task_dir, "test-crossing", "3"))]

dict_train_class_0 = dict.fromkeys(labeled_train_class_0_names, 0)
dict_train_class_1 = dict.fromkeys(labeled_train_class_1_names, 1)
dict_train_class_2 = dict.fromkeys(labeled_train_class_2_names, 2)
dict_train_class_3 = dict.fromkeys(labeled_train_class_3_names, 3)

dict_test_class_0 = dict.fromkeys(labeled_test_class_0_names, 0)
dict_test_class_1 = dict.fromkeys(labeled_test_class_1_names, 1)
dict_test_class_2 = dict.fromkeys(labeled_test_class_2_names, 2)
dict_test_class_3 = dict.fromkeys(labeled_test_class_3_names, 3)

dict_labeled_data = dict(dict_train_class_0.items() + dict_train_class_1.items() + dict_train_class_2.items() + dict_train_class_3.items() + dict_test_class_0.items() + dict_test_class_1.items() + dict_test_class_2.items() + dict_test_class_3.items())



if osp.exists(ldbpath_train):
    shutil.rmtree(ldbpath_train)
ldb_train = leveldb.LevelDB(ldbpath_train)

if osp.exists(ldbpath_test):
    shutil.rmtree(ldbpath_test)
ldb_test = leveldb.LevelDB(ldbpath_test)

if osp.exists(rawimage_path_train):
    shutil.rmtree(rawimage_path_train)
os.makedirs(rawimage_path_train)

if osp.exists(rawimage_path_test):
    shutil.rmtree(rawimage_path_test)
os.makedirs(rawimage_path_test)

h5_db = h5py.File(h5_filename)


for (demo_id, demo_name) in enumerate(h5_db.keys()):
    demo = h5_db[demo_name]
    for seg_name in demo.keys():
        seg = demo[seg_name]
        labeled_points = seg['labeled_points']
        labeled_points = np.asarray(labeled_points)
        rgb = seg['rgb']
        
        samples = []
        for i in range(labeled_points.shape[0]):
            y = labeled_points[i, 0]
            x = labeled_points[i, 1]
            cross_label = labeled_points[i, 2]
            
            if cross_label != 1 and cross_label != -1:
                continue
            
            label = 0
                
            x_start = x - int(np.floor(patch_size / 2.))
            x_end = x + int(np.floor(patch_size / 2.))
            y_start = y - int(np.floor(patch_size / 2.))
            y_end = y + int(np.floor(patch_size / 2.))
            
            if x_start >= 0 and x_end < rgb.shape[0]:
                if y_start >= 0 and y_end < rgb.shape[1]:
                    sample = rgb[x_start:x_end, y_start:y_end, :]
                    samples.append((sample, label))
                    sample_rot1 = np.rot90(sample).copy()
                    samples.append((sample_rot1, label))
                    sample_rot2 = np.rot90(sample_rot1).copy()
                    samples.append((sample_rot2, label))
                    sample_rot3 = np.rot90(sample_rot2).copy()
                    samples.append((sample_rot3, label))
                    samples.append((np.fliplr(sample).copy(), label))
                    samples.append((np.fliplr(sample_rot1).copy(), label))
                    samples.append((np.fliplr(sample_rot2).copy(), label))
                    samples.append((np.fliplr(sample_rot3).copy(), label))
                    
            
        if labeled_points.shape[0] == 0:
            print demo_name, seg_name, labeled_points.shape
            continue
        
                
        batch = leveldb.WriteBatch()        
        for (i_sample, sample) in enumerate(samples):
            patch = sample[0]
            patch_unique_id = demo_name + "_" + seg_name + "_" + str(i_sample) + ".jpg"
            label = dict_labeled_data[patch_unique_id]
            
            if demo_id < args.test_demo_start:
                cv2.imwrite(osp.join(rawimage_path_train, patch_unique_id), patch)
            else:
                cv2.imwrite(osp.join(rawimage_path_test, patch_unique_id), patch)
            
            print patch_unique_id, label
            patch = create_leveldb_utils.cv2datum(patch) 
            datum = caffe.io.array_to_datum(patch, label)
            patch_string = datum.SerializeToString()
            
            batch.Put(patch_unique_id, patch_string)
            
            
        if demo_id < args.test_demo_start:
            ldb_train.Write(batch, sync=True)
        else:
            ldb_test.Write(batch, sync=True)
        
        
            
            
            
        
        
        
        
    
