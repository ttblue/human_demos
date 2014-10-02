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
args = parser.parse_args()

def create_leveldb(data_dict, data_dir, db):
    batch = leveldb.WriteBatch()
    
    for key in data_dict.keys():
        image_filename = key
        image_label = data_dict[key]
        print image_filename, image_label
        rgb = cv2.imread(osp.join(data_dir, image_filename))
        rgb = cv2datum(rgb)
        datum = caffe.io.array_to_datum(rgb, image_label)
        batch.Put(image_filename, datum.SerializeToString())
        
    db.Write(batch, sync=True)


task_dir = osp.join(demo_files_dir, args.demo_type)
train_patches_folder = osp.join(task_dir, "trainData")
test_patches_folder = osp.join(task_dir, "testData")
train_raw_patches_folder = osp.join(task_dir, "trainData_raw")
test_raw_patches_folder = osp.join(task_dir, "testData_raw")


dict_image_class = {'background': 0, 'cornerA': 1, 'cornerB': 2, 'cornerC': 3, 'edgeA': 4,
                    'edgeB': 5, 'edgeC': 6, 'edgeAB': 7, 'edgeBC': 8, 'edgeAC': 9, 'face': 10}



train_image_classes = [f for f in listdir(train_patches_folder)]
test_image_classes = [f for f in listdir(test_patches_folder)]


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


create_leveldb(dict_train, train_raw_patches_folder, ldb_train)
create_leveldb(dict_test, test_raw_patches_folder, ldb_test)







dict_image_class_rough= {'background': 0, 'cornerA': 1, 'cornerB': 1, 'cornerC': 1, 'edgeA': 2,
                           'edgeB': 2, 'edgeC': 2, 'edgeAB': 2, 'edgeBC': 2, 'edgeAC': 2, 'face': 3}


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


create_leveldb(dict_train_rough, train_raw_patches_folder, ldb_train_rough)
create_leveldb(dict_test_rough, test_raw_patches_folder, ldb_test_rough)




        


    

    
    


