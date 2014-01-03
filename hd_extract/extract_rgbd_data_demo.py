'''
Script to extract data from rgbd+rosbag demonstration
'''
from hd_extract import extract_data as ed
import rosbag as rb
import argparse
import os, os.path as osp
import yaml
from hd_utils.defaults import demo_files_dir



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-dtype", help="type of demonstration", action='store', dest='demo_type', type=str)
    parser.add_argument("-dname", help="name of demonstration", action='store', dest='demo_name', type=str, default='')
    args = parser.parse_args()


    demo_type_dir = osp.join(demo_files_dir, args.demo_type)
    demo_master_file = osp.join(demo_type_dir, "master.yaml")
    
    with open(demo_master_file, 'r') as fh:
        demos_info = yaml.load(fh)
            
    if args.demo_name == '':
        for demo_info in demos_info["demos"]:
            ed.save_observations_rgbd(args.demo_type, demo_info["demo_name"], demo_info["calib_file"], len(demo_info["video_dirs"]))
    else:
        for demo_info in demos_info["demos"]:
            if args.demo_name == demo_info["demo_name"]:
                ed.save_observations_rgbd(args.demo_type, demo_info["demo_name"], demo_info["calib_file"], len(demo_info["video_dirs"]))
                break;
            
    print "done extraction data"