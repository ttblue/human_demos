#!/usr/bin/env python

"""
Generate hdf5 file based on master files.
Assumes that the first camera has the RGBD info.
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
from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name
from hd_utils.extraction_utils import get_video_frames
from hd_utils import clouds, color_match
from hd_utils.extraction_utils import track_video_frames_offline, track_video_frames_online
from hd_utils.defaults import demo_files_dir, hd_data_dir,\
        ar_init_dir, ar_init_demo_name, ar_init_playback_name, \
        tfm_head_dof, tfm_bf_head, cad_files_dir
import cPickle
from hd_utils.utils import avg_transform
import openravepy

from hd_utils.cloud_proc_funcs import extract_green 

import rospy
rospy.init_node('generate_h5')


parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--cloud_proc_func", default="extract_red")
parser.add_argument("--cloud_proc_mod", default="hd_utils.cloud_proc_funcs")
parser.add_argument("--no_clouds", action="store_true")
parser.add_argument("--clouds_only", action="store_true")
parser.add_argument("--verify", action="store_true")
parser.add_argument("--visualize", action="store_true", default=False)
parser.add_argument("--prompt", action="store_true", default=False)
parser.add_argument("--has_hitch", action="store_true")
parser.add_argument("--start_at", default="demo00001", type=str)
parser.add_argument("--ref_image", default="", type=str)
parser.add_argument("--offline", action="store_true")

args = parser.parse_args()

if not args.no_clouds:
    cloud_proc_mod = importlib.import_module(args.cloud_proc_mod)
    cloud_proc_func = getattr(cloud_proc_mod, args.cloud_proc_func)
    hitch_proc_func = getattr(cloud_proc_mod, "extract_hitch")

# add rgbd for one demonstration


def add_rgbd_to_hdf(video_dir, annotation, hdfroot, demo_name):
    
    if demo_name in hdfroot.keys():
        demo_group = hdfroot[demo_name]
    else:
        demo_group = hdfroot.create_group(demo_name)
        
    frame_stamps = [seg_info["look"] for seg_info in annotation]
    rgb_imgs, depth_imgs= get_video_frames(video_dir, frame_stamps)
    
    for (i_seg, seg_info) in enumerate(annotation):
        
        if seg_info["name"] == "done": continue
        
        if seg_info["name"] in demo_group.keys():
            seg_group = demo_group[seg_info["name"]]
        else:
            seg_group = demo_group.create_group(seg_info["name"])
    

        seg_group["rgb"] = rgb_imgs[i_seg]
        
        seg_group["depth"] = depth_imgs[i_seg]
        
                
        
def add_traj_to_hdf_full_demo(traj, annotation, hdfroot, demo_name):
    '''
    add traj from kalman filter (for full demo, using old kf)
    '''
     
    # get stamps of the trajectory
    for (lr, tr) in traj.items():
        stamps = tr["stamps"]
        break
                 
    demo_group = hdfroot.create_group(demo_name)
         
    for seg_info in annotation:
        if seg_info["name"] == "done": continue
        
        seg_group = demo_group.create_group(seg_info["name"]) 
         
        start = seg_info["start"]
        stop = seg_info["stop"]
         
        [i_start, i_stop] = np.searchsorted(stamps, [start, stop])
         
        traj_seg = {}
         
        for lr in traj:
            traj_seg[lr] = {}
            traj_seg[lr]["tfms"] = traj[lr]["tfms"][i_start:i_stop+1]
            traj_seg[lr]["tfms_s"] = traj[lr]["tfms_s"][i_start:i_stop+1]
            traj_seg[lr]["pot_angles"] = traj[lr]["pot_angles"][i_start:i_stop+1]
            traj_seg[lr]["stamps"] = traj[lr]["stamps"][i_start:i_stop+1]
                             
             
        for lr in traj:
            lr_group = seg_group.create_group(lr)
            lr_group["tfms"] = traj_seg[lr]["tfms"]
            lr_group["tfms_s"] = traj_seg[lr]["tfms_s"]
            lr_group["pot_angles"] = traj_seg[lr]["pot_angles"]
            lr_group["stamps"] = traj_seg[lr]["stamps"]
            
def add_traj_to_hdf(traj, annotation, hdfroot, demo_name):
    '''
    add segmented traj from kalman filter (using new kf)
    '''
    
    demo_group = hdfroot.create_group(demo_name)

    for seg_info in annotation:
        seg_name = seg_info["name"]
        if seg_name == "done": continue
        
        seg_group = demo_group.create_group(seg_name)
        
        for lr in traj:
            lr_group = seg_group.create_group(lr)
            
            lr_group["tfms"] = traj[lr][seg_name]["tfms"]
            lr_group["tfms_s"] = traj[lr][seg_name]["tfms_s"]
            lr_group["pot_angles"] = traj[lr][seg_name]["pot_angles"]
            lr_group["stamps"] = traj[lr][seg_name]["stamps"]
            lr_group["covars"] = traj[lr][seg_name]["covars"]
            #lr_group["covars_s"] = traj[lr][seg_name]["covars_s"]        


task_dir = osp.join(demo_files_dir, args.demo_type)
task_file = osp.join(task_dir, master_name)


with open(task_file, "r") as fh: task_info = yaml.load(fh)
h5path = osp.join(task_dir, task_info["h5path"].strip())



demo_type = args.demo_type # should be same as task_info['name']
if args.ref_image == "":
    ref_image = None
else:
    ref_image = cv2.imread(osp.join(task_dir, args.ref_image))
    
if not args.clouds_only:
    if osp.exists(h5path):
        os.unlink(h5path)
    hdf = h5py.File(h5path)
    
    
    demos_info = task_info['demos']
    
    
    for demo_info in demos_info:
        demo_name = demo_info['demo_name']
        print demo_name
        demo_dir = osp.join(task_dir, demo_name)
        
        rgbd_dir = osp.join(demo_dir, demo_names.video_dir%(1))
    
        annotation_file = osp.join(demo_dir,"ann.yaml")
        traj_file = osp.join(demo_dir, "demo.traj")
        
        with open(annotation_file, "r") as fh: annotations = yaml.load(fh)
        with open(traj_file, "r") as fh: traj = cp.load(fh)
        
        add_traj_to_hdf(traj, annotations, hdf, demo_name)    
    
        # assumes the first camera contains the rgbd info        
        add_rgbd_to_hdf(rgbd_dir, annotations, hdf, demo_name)
        
        
        # test tracking
        init_tfm = None
        # Get ar marker from demo:
        ar_demo_file = osp.join(hd_data_dir, ar_init_dir, ar_init_demo_name)
        with open(ar_demo_file,'r') as fh: ar_demo_tfms = cPickle.load(fh)
        # use camera 1 as default
        ar_marker_cameras = [1]
        ar_demo_tfm = avg_transform([ar_demo_tfms['tfms'][c] for c in ar_demo_tfms['tfms'] if c in ar_marker_cameras])

        # Get ar marker for PR2:
        # default demo_file
        ar_run_file = osp.join(hd_data_dir, ar_init_dir, ar_init_playback_name)
        with open(ar_run_file,'r') as fh: ar_run_tfms = cPickle.load(fh)
        ar_run_tfm = ar_run_tfms['tfm']

        # transform to move the demo points approximately into PR2's frame
        # Basically a rough transform from head kinect to demo_camera, given the tables are the same.
        init_tfm = ar_run_tfm.dot(np.linalg.inv(ar_demo_tfm))
        init_tfm = tfm_bf_head.dot(tfm_head_dof).dot(init_tfm)
        
        
        seg_info = hdf[demo_name][annotations[0]["name"]]
        if ref_image is None:
            rgb_image = np.asarray(seg_info["rgb"])
        else:
            rgb_image = color_match.match(ref_image, np.asarray(seg_info["rgb"]))
        
        #table_cloud0, table_plane0 = cloud_proc_mod.extract_table_ransac(rgb_image, np.asarray(seg_info["depth"]), np.eye(4))
        table_cloud, table_plane = cloud_proc_mod.extract_table_pca(rgb_image, np.asarray(seg_info["depth"]), np.eye(4))
                        
        
        if args.offline:
            track_video_frames_offline(rgbd_dir, np.eye(4), init_tfm, ref_image, None) #offline
        else:
            track_video_frames_online(rgbd_dir, np.eye(4), init_tfm, table_plane, ref_image, None) #online

        
else:
    hdf = h5py.File(h5path, "r+")

# now should extract point cloud
from mpl_toolkits.mplot3d import axes3d
import pylab
fig = pylab.figure()
from mayavi import mlab

started = False
if not args.no_clouds:
    
    del_fields = ["cloud_xyz", "cloud_proc_func", "cloud_proc_mod", "cloud_proc_code",\
                  "full_cloud_xyz","full_hitch","full_object","hitch","object","hitch_pos","cloud_xyz", "table_plane"]
    
    table_cloud = None
    
    for (demo_name, demo_info) in hdf.items():
        if args.start_at and demo_name == args.start_at:
            started = True

        if not started: continue

        hitch_found = False    
        for (seg_name, seg_info) in demo_info.items():
            
            if ref_image is None:
                rgb_image = np.asarray(seg_info["rgb"])
            else:
                rgb_image = color_match.match(ref_image, np.asarray(seg_info["rgb"]))
            
            if args.prompt and args.visualize:
                if "full_cloud_xyz" in seg_info:
                    xyz = seg_info["full_cloud_xyz"]
                elif "full_hitch" in seg_info and "full_object" in seg_info:
                    xyz = np.r_[seg_info["full_hitch"], seg_info["full_object"]]
                    if table_cloud is None:
                        table_cloud = cloud_proc_mod.extract_table_ransac(np.array(seg_info["full_object"]))
#                     mlab.figure(0)
#                     mlab.clf()
#                     mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2], color=(0,0,1), scale_factor=.005)

                
                fig.clf()
                ax = fig.gca(projection='3d')
                ax.set_autoscale_on(False)
                xyzm = np.mean(xyz, axis=0)

                #should be ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'o')
                #but we shift the coordinate a bit for better plotting
                ax.plot(xyz[:,0]-xyzm[0], xyz[:,1]-xyzm[1]+1.0, xyz[:,2]-xyzm[2], 'o')

                #raw_xyz = cloud_proc_mod.extract_raw_cloud(rgb_image, np.asarray(seg_info["depth"]), np.eye(4))
                #ax.plot(raw_xyz[:,0]-xyzm[0], raw_xyz[:,1]-xyzm[1]+1.0, raw_xyz[:,2]-xyzm[2], 'o', color = (0,1,1,0.5))
                if "table_plane" in seg_info:
                    table_plane = np.asarray(seg_info["table_plane"])
                    table_cloud = cloud_proc_mod.plot_plane(table_plane[0], table_plane[1], table_plane[2], table_plane[3])
                    ax.plot_surface(table_cloud[0]-xyzm[0], table_cloud[1]-xyzm[1]+1.0, table_cloud[2]-xyzm[2], color = (0,1,0,0.5))
                fig.show()
                
                
                print demo_name, seg_name
                q = raw_input("Hit c to change the pc. q to exit")
                if q == 'q':
                    hdf.close()
                    exit()
                elif q != 'c':
                    continue
                    
            print "gen point clouds: %s %s"%(demo_name, seg_name)
            
            
            
            cloud = cloud_proc_func(rgb_image, np.asarray(seg_info["depth"]), np.eye(4))
            if args.prompt and args.visualize:
#                 mlab.figure(0)
#                 mlab.clf()
#                 mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2], color=(0,0,1), scale_factor=.005)
                xyz2 = cloud
                if args.has_hitch:
                    if not hitch_found:
                        hitch_normal = clouds.clouds_plane(cloud)
                        hitch, hitch_pos = hitch_proc_func(rgb_image, np.asarray(seg_info["depth"]), np.eye(4), hitch_normal)
                        hitch_found = True
                    xyz2 = np.r_[xyz2, hitch]
                fig.clf()
                ax = fig.gca(projection='3d')
                ax.set_autoscale_on(False)
                xyzm2 = np.mean(xyz2, axis=0)
                ax.plot(xyz2[:,0]-xyzm2[0], xyz2[:,1]-xyzm2[1]+1.0, xyz2[:,2]-xyzm2[2], 'o')
                #ax.plot(xyz2[:,0], xyz2[:,1], xyz2[:,2], 'o')
                fig.show()
                print demo_name, seg_name
                print "Before", xyz.shape
                print "After", xyz2.shape
                q = raw_input("Again: Hit c to change the pc. q to quit")
                if q == 'q':
                    hdf.close()
                    exit()
                elif q != 'c':
                    continue
            
            print "Changing."
            for field in del_fields:
                try:
                    print "1",  seg_info.keys()
                    print "2", del_fields
                    if field in seg_info: del seg_info[field]
                except Exception as e:
                    import IPython
                    IPython.embed()

            if args.has_hitch:
                if not hitch_found:
                    hitch_normal = clouds.clouds_plane(cloud)
                    hitch, hitch_pos = hitch_proc_func(rgb_image, np.asarray(seg_info["depth"]), np.eye(4), hitch_normal)
                    hitch_found = True
                seg_info["full_hitch"] = hitch
                seg_info["full_object"] = cloud
                seg_info["hitch"] = clouds.downsample(hitch, .01)
                seg_info["object"] = clouds.downsample(cloud, .01)
                seg_info["hitch_pos"] = hitch_pos
                seg_info["cloud_xyz"] = np.r_[seg_info["hitch"], seg_info["object"]]
            else:
                seg_info["full_cloud_xyz"] = cloud
                seg_info["cloud_xyz"] = clouds.downsample(cloud, .01)

            seg_info["cloud_proc_func"] = args.cloud_proc_func
            seg_info["cloud_proc_mod"] = args.cloud_proc_mod
            seg_info["cloud_proc_code"] = inspect.getsource(cloud_proc_func)
            
            if table_cloud is None:
                table_cloud, table_plane = cloud_proc_mod.extract_table_ransac(rgb_image, np.asarray(seg_info["depth"]), np.eye(4))
            

            seg_info["table_plane"] = table_plane
            print table_plane
                

        
hdf.close()      
            
if args.verify:
    
    verify_dir = osp.join(task_dir, verify_name)
    
    
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
    hdf.close()
