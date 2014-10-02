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
from sets import Set

from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name


parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--save_image", action="store_true")
parser.add_argument("--train_set_percentage", type=float, default=0.8)


args = parser.parse_args()

task_dir = osp.join(demo_files_dir, args.demo_type)
h5_file = osp.join(task_dir, args.demo_type+".h5")
h5_data = h5py.File(h5_file, 'r+')

ldbpath_train = osp.join(task_dir, "leveldb-train-towel-64")
ldbpath_test = osp.join(task_dir, "leveldb-test-towel-64")

if osp.exists(ldbpath_train):
    shutil.rmtree(ldbpath_train)
ldb_train = leveldb.LevelDB(ldbpath_train)

if osp.exists(ldbpath_test):
    shutil.rmtree(ldbpath_test)
ldb_test = leveldb.LevelDB(ldbpath_test)

dict_type_label = {'background': 0, 'corner': 1, 'edge': 2, 'fold': 3, 'wrinkled interior': 4, 'flat interior': 5}


# save sampled images for test
if args.save_image:
    imagepath_train = osp.join(task_dir, "train-towel-64")
    imagepath_test = osp.join(task_dir, "test-towel-64")

    if osp.exists(imagepath_train):
        shutil.rmtree(imagepath_train)
    os.mkdir(imagepath_train)
    for type in dict_type_label:
        os.mkdir(osp.join(imagepath_train, type))

    if osp.exists(imagepath_test):
        shutil.rmtree(imagepath_test)
    os.mkdir(imagepath_test)
    for type in dict_type_label:
        os.mkdir(osp.join(imagepath_test, type))

    
    
batch_train = leveldb.WriteBatch()
batch_test = leveldb.WriteBatch()

image_indices = Set([])

dict_towel_type_img_ids = {}
for patch_id in h5_data.keys():
    img_id = str(np.array(h5_data[patch_id]['img_id']))
    
    img_id = img_id.strip("\\")
    for i in reversed(range(len(img_id))):
        if not img_id[i:].isdigit():
            towel_type = img_id[:i+1]
            break
    
    if towel_type == "wrwrinkled_interior_towel":
        towel_type = "wrinkled_interior_towel"    
    if towel_type == "folded_half_towelfolded_half_towel":
        towel_type = "folded_half_towel"
    img_id2 = towel_type + img_id[i+1:]
    
    h5_data[patch_id]['img_id'][()] = img_id2
    
    if towel_type in dict_towel_type_img_ids.keys():
        dict_towel_type_img_ids[towel_type].add(img_id2)
    else:
        dict_towel_type_img_ids[towel_type] = Set([img_id2])
    
print dict_towel_type_img_ids

train_img_ids = []
test_img_ids = []

for towel_type in dict_towel_type_img_ids:
    img_ids = dict_towel_type_img_ids[towel_type]
    img_ids = list(img_ids)
    n = np.ceil(args.train_set_percentage * len(img_ids)).astype(int)
    np.random.shuffle(img_ids)
    train_img_ids = train_img_ids + img_ids[:n]
    test_img_ids = test_img_ids + img_ids[n:]
    print n, len(img_ids) - n
    
print len(train_img_ids), len(test_img_ids) 



for patch_id in h5_data.keys():
    patch_img = np.array(h5_data[patch_id]['rgb'])
    img_id = str(np.array(h5_data[patch_id]['img_id']))
    type = str(np.array(h5_data[patch_id]['type']))
    label = dict_type_label[type]
    
    patch = cv2datum(patch_img)
    patch = caffe.io.array_to_datum(patch, label)
    patch = patch.SerializeToString()
    patch_key = patch_id
    
    if img_id in train_img_ids:
        batch_train.Put(patch_key, patch)
        if args.save_image:
            cv2.imwrite(osp.join(imagepath_train, type, patch_key+".jpg"), patch_img)
    elif img_id in test_img_ids:
        batch_test.Put(patch_key, patch)
        if args.save_image:
            cv2.imwrite(osp.join(imagepath_test, type, patch_key+".jpg"), patch_img)
    else:
        print "error happens"



ldb_train.Write(batch_train, sync=True)
ldb_test.Write(batch_test, sync=True)    
    
