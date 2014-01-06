'''
Script to extract data from rgbd+rosbag demonstration
'''
from hd_extract import extract_data as ed
import rosbag as rb
import argparse
import os, os.path as osp
import yaml
from hd_utils.defaults import demo_files_dir, demo_names, master_name



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-demo_type", help="type of demonstration", action='store', dest='demo_type', type=str)
    parser.add_argument("-demo_name", help="name of demonstration", action='store', dest='demo_name', type=str, default='')
    args = parser.parse_args()


    demo_type_dir = osp.join(demo_files_dir, args.demo_type)
    demo_master_file = osp.join(demo_type_dir, master_name)
    
    with open(demo_master_file, 'r') as fh:
        demos_info = yaml.load(fh)
            
    if args.demo_name == '':
        for demo in demos_info["demos"]:
            demo_dir = osp.join(demo_type_dir, demo["demo_name"])
            with open(osp.join(demo_dir, demo_names.camera_types_name)) as fh: cam_types = yaml.load(fh)
            ed.save_observations_rgbd(args.demo_type, demo["demo_name"], demo_names.calib_name, len(cam_types))
    else:
        if args.demo_name in (demo["demo_name"] for demo in demos_info["demos"]):
            demo_dir = osp.join(demo_type_dir, args.demo_name)
            with open(osp.join(demo_dir, demo_names.camera_types_name)) as fh: cam_types = yaml.load(fh)
            ed.save_observations_rgbd(args.demo_type, args.demo_name, demo_names.calib_name, len(cam_types))
            
    print "Done extracting data."