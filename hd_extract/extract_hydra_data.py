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
            # Wait until current demo is done recording, if so.
            while osp.isfile(osp.join(demo_dir, demo_names.record_demo_temp)):
                time.sleep(1)
            # Some other node is extracting data currently.
            if osp.isfile(osp.join(demo_dir, demo_names.extract_hydra_data_temp)): continue
            # Check if data file already exists
            if not osp.isfile(osp.join(demo_dir, demo_names.hydra_data_name)):
                ed.save_hydra_only(args.demo_type, demo["demo_name"], demo_names.calib_name)                    
            else:
                yellowprint("Hydra data file exists for %s. Not overwriting."%demo["demo_name"])
            

    else:
        if args.demo_name in (demo["demo_name"] for demo in demos_info["demos"]):
            demo_dir = osp.join(demo_type_dir, args.demo_name)
            # Wait until current demo is done recording, if so.
            while osp.isfile(osp.join(demo_dir, demo_names.record_demo_temp)):
                time.sleep(1)
            # Check if some other node is extracting data currently.
            if not osp.isfile(osp.join(demo_dir, demo_names.extract_hydra_data_temp)):
                # Check if data file already exists
                if osp.isfile(osp.join(demo_dir, demo_names.hydra_data_name)):
                    if yes_or_no('Hydra data file already exists for this demo. Overwrite?'):
                        ed.save_hydra_only(args.demo_type, args.demo_name, demo_names.calib_name)
                else:
                    ed.save_hydra_only(args.demo_type, args.demo_name, demo_names.calib_name)

    print "Done extracting hydra data."