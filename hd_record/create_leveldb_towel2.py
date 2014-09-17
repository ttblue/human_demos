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
parser.add_argument("--num_samples", type=int)
parser.add_argument("--sample_patch_size", type=int)
parser.add_argument("--test_demo_start", type=int, default=13)
parser.add_argument("--save_image", action="store_true")
parser.add_argument("--background_accept_ratio", type=float, default=0.05)

args = parser.parse_args()
patch_size = args.sample_patch_size

task_dir = osp.join(demo_files_dir, args.demo_type)

ldbpath_train = osp.join(task_dir, "leveldb-train-rand-towel-"+str(patch_size))
ldbpath_test = osp.join(task_dir, "leveldb-test-rand-towel-"+str(patch_size))

if osp.exists(ldbpath_train):
    shutil.rmtree(ldbpath_train)
ldb_train = leveldb.LevelDB(ldbpath_train)

if osp.exists(ldbpath_test):
    shutil.rmtree(ldbpath_test)
ldb_test = leveldb.LevelDB(ldbpath_test)

# save sampled images for test
if args.save_image:
    imagepath_train = osp.join(task_dir, "train-rand-towel-"+str(patch_size))
    imagepath_test = osp.join(task_dir, "test-rand-towel-"+str(patch_size))

    if osp.exists(imagepath_train):
        shutil.rmtree(imagepath_train)
    os.mkdir(imagepath_train)

    if osp.exists(imagepath_test):
        shutil.rmtree(imagepath_test)
    os.mkdir(imagepath_test)

imagepath = osp.join(task_dir, "towel-image")
image_indices = listdir(imagepath)
image_indices.sort()


print image_indices

num_images = len(image_indices)
batch_train = leveldb.WriteBatch()
batch_test = leveldb.WriteBatch()

print num_images



#for i in range(args.num_samples):
i = 0
while i < args.num_samples:
    image_id = i % num_images
    demo_name = image_indices[image_id]
        
    image = cv2.imread(osp.join(imagepath, demo_name))
    
    (h, w, c) = image.shape
        
    start_x = np.random.randint(0, h - patch_size)
    #start_x = np.random.randint(190, h - patch_size)
    start_y = np.random.randint(0, w - patch_size)
    #start_y = np.random.randint(130, 540 - patch_size)    
    
    patch_img = image[start_x:start_x + patch_size, start_y:start_y + patch_size, :]
    patch_img2 = patch_img.reshape(patch_img.shape[0] * patch_img.shape[1], patch_img.shape[2])
    patch_var = np.var(patch_img2, 0)
    if patch_var[0] > 1000 or patch_var[1] > 1000 or patch_var[2] > 1000:
        pass
    elif patch_var[0] < 100 and patch_var[1] < 100 and patch_var[2] < 100:
        rnd_v = np.random.uniform(0, 1);
        if rnd_v > args.background_accept_ratio:
            continue
    else: 
        rnd_v = np.random.uniform(0, 1)
        if rnd_v > 0.5:
            continue
    

    print i, np.var(patch_img2, 0)
    patch = cv2datum(patch_img)
    patch = caffe.io.array_to_datum(patch)
    patch = patch.SerializeToString()
    
    
    patch_key = str(i)
    
    if image_id < args.test_demo_start:
        batch_train.Put(patch_key, patch)
        #cv2.imwrite(osp.join(imagepath_train, patch_key+".jpg"), patch_img)
    else:
        batch_test.Put(patch_key, patch)
        #cv2.imwrite(osp.join(imagepath_test, patch_key+".jpg"), patch_img)
        
    if args.save_image:
        if image_id < args.test_demo_start:
            cv2.imwrite(osp.join(imagepath_train, patch_key+".jpg"), patch_img)
        else:
            cv2.imwrite(osp.join(imagepath_test, patch_key+".jpg"), patch_img)
            
    i = i + 1
        
ldb_train.Write(batch_train, sync=True)
ldb_test.Write(batch_test, sync=True)