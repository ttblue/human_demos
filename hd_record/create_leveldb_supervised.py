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


parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--patch_size", default=64, type=int)
parser.add_argument("--apply_ZCA", action="store_true")
parser.add_argument("--ZCA_file", default='', type=str)
args = parser.parse_args()


def create_leveldb(data_dict, data_dir, db):
    batch = leveldb.WriteBatch()
    
    for key in data_dict.keys():
        image_filename = key
        image_label = data_dict[key]
        rgb = cv2.imread(osp.join(data_dir, image_filename))
        rgb = cv2datum(rgb)
        datum = caffe.io.array_to_datum(rgb, image_label)
        batch.Put(image_filename, datum.SerializeToString())
        
    db.Write(batch, sync=True)


task_dir = osp.join(demo_files_dir, args.demo_type)
train_patches_folder = osp.join(task_dir, "trainData_raw")
test_patches_folder= osp.join(task_dir, "testData_raw")
train_positive_patches_folder = osp.join(task_dir, "trainData")
test_positive_patches_folder = osp.join(task_dir, "testData")

train_image_names = [f for f in listdir(train_patches_folder)]
test_image_names = [f for f in listdir(test_patches_folder)]
train_positive_image_names = [f for f in listdir(train_positive_patches_folder)]
test_positive_image_names = [f for f in listdir(test_positive_patches_folder)]

dict_train = dict.fromkeys(train_image_names, 0)
dict_test= dict.fromkeys(test_image_names, 0)

dict_train_positive = dict.fromkeys(train_positive_image_names, 1)
dict_test_positive= dict.fromkeys(test_positive_image_names, 1)


for key in train_positive_image_names:
    dict_train[key] = 1

for key in test_positive_image_names:
    dict_test[key] = 1
    
ldbpath_train = osp.join(task_dir, "leveldb-train-labeled-ZCA")
ldbpath_test = osp.join(task_dir, "leveldb-test-labeled-ZCA")

ldbpath_train_positive = osp.join(task_dir, "leveldb-train-positive-ZCA")
ldbpath_test_positive = osp.join(task_dir, "leveldb-test-positive-ZCA")


if osp.exists(ldbpath_train):
    shutil.rmtree(ldbpath_train)
ldb_train = leveldb.LevelDB(ldbpath_train)

if osp.exists(ldbpath_test):
    shutil.rmtree(ldbpath_test)
ldb_test = leveldb.LevelDB(ldbpath_test)

if osp.exists(ldbpath_train_positive):
    shutil.rmtree(ldbpath_train_positive)
ldb_train_positive = leveldb.LevelDB(ldbpath_train_positive)

if osp.exists(ldbpath_test_positive):
    shutil.rmtree(ldbpath_test_positive)
ldb_test_positive = leveldb.LevelDB(ldbpath_test_positive)

create_leveldb(dict_train, train_patches_folder, ldb_train)
create_leveldb(dict_test, test_patches_folder, ldb_test)

create_leveldb(dict_train_positive, train_positive_patches_folder, ldb_train_positive)
create_leveldb(dict_test_positive, test_positive_patches_folder, ldb_test_positive)

#if args.apply_ZCA:
#    print "ZCA whitening"
#    X_train = collect_data_from_leveldb(ldb_train)
#    X_test = collect_data_from_leveldb(ldb_test)
#    
#    if args.ZCA_file == '':
#        ZCA_mean, ZCA_rot = compute_ZCA(X_train)
#        cp.dump([ZCA_mean, ZCA_rot], open("ZCA_leveldb-train-labeled.cp"))
#    else:
#        ZCA_mean, ZCA_rot = cp.load(open(args.ZCA_file))
#        
#    apply_ZCA_db(ldb_train, ZCA_mean, ZCA_rot)
#    apply_ZCA_db(ldb_test, ZCA_mean, ZCA_rot)


if args.apply_ZCA:
    print "ZCA whitening"
    #X_train = collect_data_from_leveldb(ldb_train)
    
    #compute_ZCA_fast(X_train, args.ZCA_file)
    apply_ZCA_db_fast(ldb_train, args.ZCA_file)
    apply_ZCA_db_fast(ldb_test, args.ZCA_file)
    
    
    X_train_positive = collect_data_from_leveldb(ldb_train_positive)
    compute_ZCA_fast(X_train_positive, 'zcap')
    apply_ZCA_db_fast(ldb_train_positive, 'zcap')
    apply_ZCA_db_fast(ldb_test_positive, 'zcap')
        
    





    




