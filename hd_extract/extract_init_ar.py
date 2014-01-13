'''
Script to extract hydra data from rosbag of demonstration.
'''
import argparse
import os.path as osp
import yaml
import time

from hd_utils.defaults import demo_files_dir, demo_names, master_name
from hd_utils.colorize import yellowprint
from hd_utils.yes_or_no import yes_or_no

from hd_extract import extract_data as ed


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_type", help="type of demonstration", action='store', type=str)
    parser.add_argument("--demo_name", help="name of demonstration", action='store', type=str)
    parser.add_argument("--ar_marker", help="number of ar marker to find", action='store', type=int)
    args = parser.parse_args()


    demo_type_dir = osp.join(demo_files_dir, args.demo_type)
    demo_master_file = osp.join(demo_type_dir, master_name)
    demo_dir = osp.join(demo_type_dir, args.demo_name)
    
    with open(demo_master_file, 'r') as fh:
        demos_info = yaml.load(fh)
            
    if args.demo_name in (demo["demo_name"] for demo in demos_info["demos"]):
        # Check if data file already exists
        if osp.isfile(osp.join(demo_dir, demo_names.init_ar_marker_name)):
            if yes_or_no('Init ar file already exists for this demo. Overwrite?'):
                ed.save_init_ar(args.demo_type, args.demo_name, args.ar_marker)
        else:
            ed.save_init_ar(args.demo_type, args.demo_name, args.ar_marker)

    print "Done extracting init ar marker."