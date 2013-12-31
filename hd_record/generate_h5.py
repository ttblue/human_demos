#!/usr/bin/env python

"""
Generate hdf5 file based on master files
"""
import os, os.path as osp
import rosbag
import h5py
import hd_utils
import yaml
import importlib, inspect
import numpy as np
import cPickle as cp
import cv2
import shutil
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("task_file")
parser.add_argument("--cloud_proc_func", default="extract_red")
parser.add_argument("--cloud_proc_mod", default="hd_utils.cloud_proc_funs")
parser.add_argument("--no_clouds")
parser.add_argument("--clouds_only", action="store_true")
parser.add_argument("--verify", action="store_true")
args = parser.parse_args()



cloud_proc_mod = importlib.import_module(args.cloud_proc_mod)
cloud_proc_func = getattr(cloud_proc_mod, args.cloud_proc_func)


def get_video_frames(video_dir, frame_stamps):
    
    video_stamps = np.loadtxt(osp.join(video_dir,"stamps.txt"))
    frame_inds = np.searchsorted(video_stamps, frame_stamps)
    
    rgbs = []
    depths = []
    for frame_ind in frame_inds:
        rgb = cv2.imread(osp.join(video_dir,"rgb%05i.jpg"%frame_ind))
        assert rgb is not None
        rgbs.append(rgb)
        depth = cv2.imread(osp.join(video_dir,"depth%05i.png"%frame_ind),2)
        assert depth is not None
        depths.append(depth)
    return rgbs, depths


# add rgbd for one demonstration
def add_rgbd_to_hdf(video_dir, annotations, hdfroot, demo_name):
    
    for (i_demo, demo_annotations) in enumerate(annotations):
        
        demo_group = hdfroot[demo_name + "_%i"%(i_demo)]
        
        frame_stamps = [seg_info["look"] for seg_info in demo_annotations]
        rgb_imgs, depth_imgs= get_video_frames(video_dir, frame_stamps)
        
        for (i_seg, seg_info) in enumerate(demo_annotations):
            seg_group = demo_group[seg_info["name"]]
            seg_group["rgb"] = rgb_imgs[i_seg]
            seg_group["depth"] = depth_imgs[i_seg]
        
        
def add_traj_to_hdf(trajs, annotations, hdfroot, demo_name):
    
    # get stamps of the trajectory
    for (lr, traj) in trajs.items():
        stamps = traj["stamps"]
        break
        
    for (i_demo, demo_annotations) in enumerate(annotations):
        demo_group = hdfroot.create_group(demo_name + "_%i"%(i_demo))
        
        for seg_info in demo_annotations:
            seg_group = demo_group.create_group(seg_info["name"]) 
            
            start = seg_info["start"]
            stop = seg_info["stop"]
            
            [i_start, i_stop] = np.searchsorted(stamps, [start, stop])
            
            traj_seg = {}
            
            for lr in trajs:
                traj_seg[lr] = {}
                traj_seg[lr]["tfms"] = trajs[lr]["tfms"][i_start:i_stop+1]
                traj_seg[lr]["tfms_s"] = trajs[lr]["tfms_s"][i_start:i_stop+1]
                traj_seg[lr]["pot_angles"] = trajs[lr]["pot_angles"][i_start:i_stop+1]
                traj_seg[lr]["stamps"] = trajs[lr]["stamps"][i_start:i_stop+1]
                
                
                
                
                
            for lr in trajs:
                lr_group = seg_group.create_group(lr)
                lr_group["tfms"] = traj_seg[lr]["tfms"]
                lr_group["tfms_s"] = traj_seg[lr]["tfms_s"]
                lr_group["pot_angles"] = traj_seg[lr]["pot_angles"]
                lr_group["stamps"] = traj_seg[lr]["stamps"]
        




task_dir = osp.dirname(args.task_file)

with open(args.task_file, "r") as fh: task_info = yaml.load(fh)
h5path = osp.join(task_dir, task_info["h5path"].strip())




if args.clouds_only:
    hdf = h5py.File(h5path, "r+")
else:
    if osp.exists(h5path):
        os.unlink(h5path)
    hdf = h5py.File(h5path)
    
    bag_info = task_info["bags"]
        
    bag_file = osp.join(task_dir, bag_info["bag_file"])
    ann_file = osp.join(task_dir, bag_info["annotation_file"])
    video_dirs = bag_info["video_dirs"]
    traj_file = osp.join(task_dir, bag_info["traj_file"])
    data_file = osp.join(task_dir, bag_info["data_file"])
            
    demo_name = bag_info["demo_name"] if "demo_name" in bag_info else "demo%i"%i_bag
            
    bag = rosbag.Bag(bag_file)
    with open(ann_file, "r") as fh: annotations = yaml.load(fh)
            
    with open(traj_file, "r") as fh: trajs = cp.load(fh)
            
            
    add_traj_to_hdf(trajs, annotations, hdf, demo_name)    
    
    # assumes the first camera contains the rgbd info        
    add_rgbd_to_hdf(osp.join(task_dir, video_dirs[0]), annotations, hdf, demo_name)

    
    
# now should extract point cloud
if not args.no_clouds:
    for (demo_name, demo_info) in hdf.items():
        
        for (seg_name, seg_info) in demo_info.items():
        
            for field in ["cloud_xyz", "cloud_proc_func", "cloud_proc_mod", "cloud_proc_code"]:
                if field in seg_info: del seg_info[field]
            
            seg_info["cloud_xyz"] = cloud_proc_func(np.asarray(seg_info["rgb"]), np.asarray(seg_info["depth"]))
    
            seg_info["cloud_proc_func"] = args.cloud_proc_func
            seg_info["cloud_proc_mod"] = args.cloud_proc_mod
            seg_info["cloud_proc_code"] = inspect.getsource(cloud_proc_func)
            
            
if args.verify:
    
    verify_dir = osp.join(task_dir, "verify")
    
    
    hdf = h5py.File(h5path, "r")
    
    for (demo_name, demo_info) in hdf.items():
        
        demo_verify_dir = osp.join(verify_dir, demo_name)
        
        if osp.exists(demo_verify_dir):
            shutil.rmtree(demo_verify_dir)
        os.makedirs(demo_verify_dir)
        
        for (seg_name, seg_info) in demo_info.items():
            
                print demo_name, seg_name
            
                rgb = np.asarray(seg_info["rgb"])
                depth = np.asarray(seg_info["depth"])
                
                
                rgb_filename = osp.basename(seg_name) + ".jpg"
                rgb_filename = osp.join(demo_verify_dir, rgb_filename)
                cv2.imwrite(rgb_filename, rgb)
                
                depth_filename = osp.basename(seg_name) + ".png"
                depth_filename = osp.join(demo_verify_dir, depth_filename)
                cv2.imwrite(depth_filename, depth)
                
                
                traj_filename = osp.basename(seg_name)
                traj_filename = osp.join(demo_verify_dir, traj_filename)
                traj_fd = open(traj_filename, "w")
                
                traj = {}
                if "l" in seg_info.keys():
                    traj["l"] = {"tfms": np.asarray(seg_info["l"]["tfms"]),
                                 "tfms_s": np.asarray(seg_info["l"]["tfms_s"]),
                                 "pot_angles": np.asarray(seg_info["l"]["pot_angles"]),
                                 "stamps": np.asarray(seg_info["l"]["stamps"])}
                
                if "r" in seg_info.keys():
                    traj["r"] = {"tfms": np.asarray(seg_info["r"]["tfms"]),
                                 "tfms_s": np.asarray(seg_info["r"]["tfms_s"]),
                                 "pot_angles": np.asarray(seg_info["r"]["pot_angles"]),
                                 "stamps": np.asarray(seg_info["r"]["stamps"])}                    
                

                
                cp.dump(traj, traj_fd)
                
            
    
    
    
    
    
    
