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

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--train_folder")
parser.add_argument("--test_folder")
args = parser.parse_args()

def create_leveldb(data_dict, data_folder_dict, map_label_to_folder, data_dir, db):
    batch = leveldb.WriteBatch()
    
    for key in data_dict.keys():
        image_filename = key
        image_label = data_dict[key]
        folder_label = data_folder_dict[key]
        image_folder = map_label_to_folder[folder_label]
        print image_filename, image_label
        rgb = cv2.imread(osp.join(data_dir, image_folder, image_filename))
        rgb = cv2datum(rgb)
        datum = caffe.io.array_to_datum(rgb, image_label)
        batch.Put(image_filename, datum.SerializeToString())
        
    db.Write(batch, sync=True)


task_dir = osp.join(demo_files_dir, args.demo_type)
train_patches_folder = osp.join(task_dir, args.train_folder)
test_patches_folder = osp.join(task_dir, args.test_folder)


dict_image_class = {'background': 0, 'corner1': 1, 'corner2': 2, 'edge1': 3, 'edge2': 4, 'face': 5}
dict_image_class_folder = {0: 'background', 1: 'corner1', 2: 'corner2', 3: 'edge1', 4: 'edge2', 5: 'face'}



train_image_classes = [f for f in listdir(train_patches_folder)]
test_image_classes = [f for f in listdir(test_patches_folder)]
if any(x for x in train_image_classes if x == 'unlabeled'):
    train_image_classes.remove('unlabeled')
if any(x for x in test_image_classes if x == 'unlabeled'):
    test_image_classes.remove('unlabeled')
    


dict_train = {}
dict_test = {}

for image_class in train_image_classes:
    label = dict_image_class[image_class]
    image_dir = osp.join(train_patches_folder, image_class)
    image_names = [f for f in listdir(image_dir)]
    for image_name in image_names:
        dict_train[image_name] = label
        
for image_class in test_image_classes:
    label = dict_image_class[image_class]
    image_dir = osp.join(test_patches_folder, image_class)
    image_names = [f for f in listdir(image_dir)]
    for image_name in image_names:
        dict_test[image_name] = label
        
ldbpath_train = osp.join(task_dir, "leveldb-train-labeled")
ldbpath_test = osp.join(task_dir, "leveldb-test-labeled")

        
if osp.exists(ldbpath_train):
    shutil.rmtree(ldbpath_train)
ldb_train = leveldb.LevelDB(ldbpath_train)

if osp.exists(ldbpath_test):
    shutil.rmtree(ldbpath_test)
ldb_test = leveldb.LevelDB(ldbpath_test)


create_leveldb(dict_train, dict_train, dict_image_class_folder, train_patches_folder, ldb_train)
create_leveldb(dict_test, dict_test, dict_image_class_folder, test_patches_folder, ldb_test)





dict_image_class_rough = {'background': 0, 'corner1': 1, 'corner2': 1, 'edge1': 2, 'edge2': 2, 'face': 3}


dict_train_rough = {}
dict_test_rough = {}

for image_class in train_image_classes:
    label = dict_image_class_rough[image_class]
    image_dir = osp.join(train_patches_folder, image_class)
    image_names = [f for f in listdir(image_dir)]
    for image_name in image_names:
        dict_train_rough[image_name] = label
        
for image_class in test_image_classes:
    label = dict_image_class_rough[image_class]
    image_dir = osp.join(test_patches_folder, image_class)
    image_names = [f for f in listdir(image_dir)]
    for image_name in image_names:
        dict_test_rough[image_name] = label



ldbpath_train_rough = osp.join(task_dir, "leveldb-train-labeled-rough")
ldbpath_test_rough = osp.join(task_dir, "leveldb-test-labeled-rough")

        
if osp.exists(ldbpath_train_rough):
    shutil.rmtree(ldbpath_train_rough)
ldb_train_rough = leveldb.LevelDB(ldbpath_train_rough)

if osp.exists(ldbpath_test_rough):
    shutil.rmtree(ldbpath_test_rough)
ldb_test_rough = leveldb.LevelDB(ldbpath_test_rough)


create_leveldb(dict_train_rough, dict_train, dict_image_class_folder, train_patches_folder, ldb_train_rough)
create_leveldb(dict_test_rough, dict_test, dict_image_class_folder, test_patches_folder, ldb_test_rough)




        


    

    
    


