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

from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name
import create_leveldb_utils

import scipy.ndimage


parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--patch_size", default=64, type=int)
parser.add_argument("--test_demo_start", type=int)
parser.add_argument("--num_bg_sample_per_region", type=int, default=1)


args = parser.parse_args()

patch_size = args.patch_size

        
task_dir = osp.join(demo_files_dir, args.demo_type)
h5_filename = osp.join(task_dir, args.demo_type + "_crossing.h5")

ldbpath_train = osp.join(task_dir, "leveldb-train-" + str(patch_size))
ldbpath_test = osp.join(task_dir, "leveldb-test-" + str(patch_size))

rawimage_path_train = osp.join(task_dir, "train-" + str(patch_size))
rawimage_path_test = osp.join(task_dir, "test-" + str(patch_size))

if osp.exists(ldbpath_train):
    shutil.rmtree(ldbpath_train)
ldb_train = leveldb.LevelDB(ldbpath_train)

if osp.exists(ldbpath_test):
    shutil.rmtree(ldbpath_test)
ldb_test = leveldb.LevelDB(ldbpath_test)

if osp.exists(rawimage_path_train):
    shutil.rmtree(rawimage_path_train)
os.makedirs(rawimage_path_train)
os.makedirs(osp.join(rawimage_path_train, "background"))
os.makedirs(osp.join(rawimage_path_train, "normal"))
os.makedirs(osp.join(rawimage_path_train, "endpoint"))
os.makedirs(osp.join(rawimage_path_train, "crossing"))




if osp.exists(rawimage_path_test):
    shutil.rmtree(rawimage_path_test)
os.makedirs(rawimage_path_test)
os.makedirs(osp.join(rawimage_path_test, "background"))
os.makedirs(osp.join(rawimage_path_test, "normal"))
os.makedirs(osp.join(rawimage_path_test, "endpoint"))
os.makedirs(osp.join(rawimage_path_test, "crossing"))

h5_db = h5py.File(h5_filename)


dict_label_to_folder = {0: "background", 1: "endpoint", 2: "normal", 3: "crossing"}


def resize_image(image, size_threshold):    
    h, w, c = image.shape
    if h < size_threshold:
        ratio = np.ceil(float(size_threshold) / h).astype(int)
        image = scipy.ndimage.zoom(image, (ratio, ratio, 1))
        
    return image

    

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
            
            if i == 0:
                label = 1
            elif i == labeled_points.shape[0] - 1:
                label = 1
            elif cross_label == 1 or cross_label == -1:
                label = 3
            else:
                label = 2
                
            x_start = x - int(np.floor(patch_size / 2.))
            x_end = x + int(np.floor(patch_size / 2.))
            y_start = y - int(np.floor(patch_size / 2.))
            y_end = y + int(np.floor(patch_size / 2.))
            
            if x_start >= 0 and x_end < rgb.shape[0]:
                if y_start >= 0 and y_end < rgb.shape[1]:
                    sample = rgb[x_start:x_end, y_start:y_end, :]
                    
                    sample = resize_image(sample, 64)
                    
                    samples.append((sample, label))
                    
                    
                    if label == 3:
                        perturb_delta = patch_size / 4
                        delta_x = perturb_delta
                        delta_y = perturb_delta
                        
                        if x_start+delta_x >= 0 and x_end+delta_x < rgb.shape[0] and y_start+delta_y >= 0 and y_end+delta_y < rgb.shape[1]:
                            sample = rgb[x_start+perturb_delta:x_end+perturb_delta, y_start+perturb_delta:y_end+perturb_delta, :]
                            sample = resize_image(sample, 64)
                            samples.append((sample, label))
                            
                        if x_start+delta_x >= 0 and x_end+delta_x < rgb.shape[0] and y_start-delta_y >= 0 and y_end-delta_y < rgb.shape[1]:
                            sample = rgb[x_start+perturb_delta:x_end+perturb_delta, y_start-perturb_delta:y_end-perturb_delta, :]
                            sample = resize_image(sample, 64)
                            samples.append((sample, label))
                    
                        if x_start-delta_x >= 0 and x_end-delta_x < rgb.shape[0] and y_start+delta_y >= 0 and y_end+delta_y < rgb.shape[1]:
                            sample = rgb[x_start-perturb_delta:x_end-perturb_delta, y_start+perturb_delta:y_end+perturb_delta, :]
                            sample = resize_image(sample, 64)
                            samples.append((sample, label))
                            
                        if x_start-delta_x >= 0 and x_end-delta_x < rgb.shape[0] and y_start-delta_y >= 0 and y_end-delta_y < rgb.shape[1]:
                            sample = rgb[x_start-perturb_delta:x_end-perturb_delta, y_start-perturb_delta:y_end-perturb_delta, :]
                            sample = resize_image(sample, 64)
                            samples.append((sample, label))                    
                    
#                    sample_rot1 = np.rot90(sample).copy()
#                    samples.append((sample_rot1, label))
#                    sample_rot2 = np.rot90(sample_rot1).copy()
#                    samples.append((sample_rot2, label))
#                    sample_rot3 = np.rot90(sample_rot2).copy()
#                    samples.append((sample_rot3, label))
#                    samples.append((np.fliplr(sample).copy(), label))
#                    samples.append((np.fliplr(sample_rot1).copy(), label))
#                    samples.append((np.fliplr(sample_rot2).copy(), label))
#                    samples.append((np.fliplr(sample_rot3).copy(), label))
            
        if labeled_points.shape[0] == 0:
            print demo_name, seg_name, labeled_points.shape
            continue
        
        ## add (at most) four samples for background
        ## first, bound for the rope
        upper_bound = np.max(labeled_points[:, :2], axis=0)
        lower_bound = np.min(labeled_points[:, :2], axis=0)
        
        if lower_bound[0] >= patch_size:
            bg_samples = create_leveldb_utils.sample_patches(rgb[:lower_bound[0], :, :], patch_size, args.num_bg_sample_per_region)
            for sample in bg_samples:
                sample = resize_image(sample, 64)
                samples.append((sample, 0))
                
#                sample_rot1 = np.rot90(sample).copy()
#                samples.append((sample_rot1, 0))
#                sample_rot2 = np.rot90(sample_rot1).copy()
#                samples.append((sample_rot2, 0))
#                sample_rot3 = np.rot90(sample_rot2).copy()
#                samples.append((sample_rot3, 0))
#                samples.append((np.fliplr(sample).copy(), 0))
#                samples.append((np.fliplr(sample_rot1).copy(), 0))
#                samples.append((np.fliplr(sample_rot2).copy(), 0))
#                samples.append((np.fliplr(sample_rot3).copy(), 0))
        
        if upper_bound[0] + 1 < rgb.shape[0] - patch_size:
            bg_samples = create_leveldb_utils.sample_patches(rgb[upper_bound[0] + 1:, :, :], patch_size, args.num_bg_sample_per_region)
            for sample in bg_samples:
                sample = resize_image(sample, 64)
                samples.append((sample, 0))
                
#                sample_rot1 = np.rot90(sample).copy()
#                samples.append((sample_rot1, 0))
#                sample_rot2 = np.rot90(sample_rot1).copy()
#                samples.append((sample_rot2, 0))
#                sample_rot3 = np.rot90(sample_rot2).copy()
#                samples.append((sample_rot3, 0))
#                samples.append((np.fliplr(sample).copy(), 0))
#                samples.append((np.fliplr(sample_rot1).copy(), 0))
#                samples.append((np.fliplr(sample_rot2).copy(), 0))
#                samples.append((np.fliplr(sample_rot3).copy(), 0))
        
        if lower_bound[1] >= patch_size:
            bg_samples = create_leveldb_utils.sample_patches(rgb[:, :lower_bound[1], :], patch_size, args.num_bg_sample_per_region)
            for sample in bg_samples:
                sample = resize_image(sample, 64)
                samples.append((sample, 0))
                
#                sample_rot1 = np.rot90(sample).copy()
#                samples.append((sample_rot1, 0))
#                sample_rot2 = np.rot90(sample_rot1).copy()
#                samples.append((sample_rot2, 0))
#                sample_rot3 = np.rot90(sample_rot2).copy()
#                samples.append((sample_rot3, 0))
#                samples.append((np.fliplr(sample).copy(), 0))
#                samples.append((np.fliplr(sample_rot1).copy(), 0))
#                samples.append((np.fliplr(sample_rot2).copy(), 0))
#                samples.append((np.fliplr(sample_rot3).copy(), 0))
                
        if upper_bound[1] + 1 < rgb.shape[1] - patch_size:
            bg_samples = create_leveldb_utils.sample_patches(rgb[:, upper_bound[1] + 1:, :], patch_size, args.num_bg_sample_per_region)
            for sample in bg_samples:
                sample = resize_image(sample, 64)
                samples.append((sample, 0))
                
#                sample_rot1 = np.rot90(sample).copy()
#                samples.append((sample_rot1, 0))
#                sample_rot2 = np.rot90(sample_rot1).copy()
#                samples.append((sample_rot2, 0))
#                sample_rot3 = np.rot90(sample_rot2).copy()
#                samples.append((sample_rot3, 0))
#                samples.append((np.fliplr(sample).copy(), 0))
#                samples.append((np.fliplr(sample_rot1).copy(), 0))
#                samples.append((np.fliplr(sample_rot2).copy(), 0))
#                samples.append((np.fliplr(sample_rot3).copy(), 0))
                
                
        batch = leveldb.WriteBatch()        
        for (i_sample, sample) in enumerate(samples):
            patch = sample[0]
            label = sample[1]
            
            patch_unique_id = demo_name + "_" + seg_name + "_" + str(i_sample)
                        
            if demo_id < args.test_demo_start:
                cv2.imwrite(osp.join(rawimage_path_train, dict_label_to_folder[label], patch_unique_id + ".jpg"), patch)
            else:
                cv2.imwrite(osp.join(rawimage_path_test, dict_label_to_folder[label], patch_unique_id + ".jpg"), patch)
            
            patch = create_leveldb_utils.cv2datum(patch) 
            datum = caffe.io.array_to_datum(patch, label)
            patch_string = datum.SerializeToString()
            
            batch.Put(patch_unique_id, patch_string)
            
            
        if demo_id < args.test_demo_start:
            ldb_train.Write(batch, sync=True)
        else:
            ldb_test.Write(batch, sync=True)
        
        
            
            
            
        
        
        
        
    
