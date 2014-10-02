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

import caffe
import leveldb
from caffe.io import caffe_pb2
import numpy as np
from scipy.misc import imsave

from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name


def uniform_sample_patches(image, patch_size, step_size):
    [nx, ny, nc] = image.shape
    sample_dim1 = int((nx - patch_size) / float(step_size) + 1)
    sample_dim2 = int((ny - patch_size) / float(step_size) + 1)
    
    if ((nx - patch_size) % step_size) != 0:
        sample_dim1 -= 1 
    
    if ((ny - patch_size) % step_size) != 0:
        sample_dim2 -= 1
        
    patches = {}
    
    for i in range(sample_dim1):
        for j in range(sample_dim2):
            patch = image[step_size * i : step_size * i + patch_size, step_size * j : step_size * j + patch_size, :]
            patches[(step_size * i, step_size * j)] = patch
    
    return patches

def rope_detection_per_image(image, net, patch_size, step_size):
    params = [v.data.shape for k, v in net.blobs.items()]
    n_parallel = params[0][0]
    #print n_parallel
    
    patches = uniform_sample_patches(image, patch_size, step_size)
    
    patches_imgs = patches.values()
    patches_starts = patches.keys()
    
    n_patch_images = len(patches)
    n_iterations = np.ceil(n_patch_images / np.double(n_parallel))
    n_iterations = int(n_iterations)
    
    rope_patch_indices = []
    for i in range(n_iterations):
        start_id = n_parallel * i
        end_id = min(n_parallel * (i + 1), n_patch_images)
        scores = net.predict(patches_imgs[start_id:end_id], oversample=True)
        if end_id == n_patch_images:
            scores = scores[:end_id-start_id, :]
    
        rope_indices= np.where((scores[:, 0] > scores[:, 1]) == 0)

        indices = range(start_id, end_id)
        indices = np.asarray(indices)
        indices = indices[rope_indices]
        rope_patch_indices.extend(indices)
    
    print rope_patch_indices  
    mask_image = np.zeros([image.shape[0], image.shape[1]])
    
    for index in rope_patch_indices:
        patch_start = patches_starts[index]
        mask_image[patch_start[0]:patch_start[0] + patch_size, patch_start[1]:patch_start[1] + patch_size] = 255
        
    return mask_image    
    
        
        

    
    

    
        
    

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--patch_size", default=64, type=int)
parser.add_argument("--protofile", type=str)
parser.add_argument("--networkfile", type=str)
parser.add_argument("--meanfile", type=str)
parser.add_argument("--step_size", default=32, type=int)
args = parser.parse_args()


task_dir = osp.join(demo_files_dir, args.demo_type)
task_file = osp.join(task_dir, master_name)
with open(task_file, "r") as fh: task_info = yaml.load(fh)


net = caffe.Classifier(args.protofile, args.networkfile)
net.set_phase_test()
net.set_mode_gpu()
net.set_mean('data', np.load(args.meanfile))
net.set_raw_scale('data', 255)
net.set_channel_swap('data', (2,1,0))

demos_info = task_info['demos']

for (index, demo_info) in enumerate(demos_info):
    demo_name = demo_info['demo_name']
    print demo_name
    demo_dir = osp.join(task_dir, demo_name)
    video_dir = osp.join(demo_dir, demo_names.video_dir%(1))
    
    mask_dir = osp.join(video_dir, 'mask')
    if not osp.exists(mask_dir):
        os.makedirs(mask_dir)

    
    video_stamps = np.loadtxt(osp.join(video_dir,demo_names.stamps_name))  
    
    
    for (index, stamp) in zip(range(len(video_stamps)), video_stamps):        
        #print index, stamp
        rgb = caffe.io.load_image(osp.join(video_dir,demo_names.rgb_name%index))
        mask_image = rope_detection_per_image(rgb, net, args.patch_size, args.step_size)
        imsave(osp.join(mask_dir, demo_names.rgb_name%index), mask_image)









