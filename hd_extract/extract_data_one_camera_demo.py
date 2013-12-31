'''
Script to extract data from rosbag demonstration
'''

from hd_track import extract_data as ed
import rosbag as rb
import argparse
import os, os.path as osp




data_dir = os.getenv('HD_DATA_DIR')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-bname", help="name of bag file", action='store', dest='bag_fname', type=str)
    parser.add_argument("-clib", help="name of calibration file", action='store', dest='calib_fname', type=str)
    parser.add_argument("-sname", help="name of save file", action='store', dest='save_fname', default='', type=str)
    vals = parser.parse_args()
    

    bag_fname = osp.join(data_dir, 'demos', vals.bag_fname)
    calib_fname = osp.join(data_dir, 'calib', vals.calib_fname)
        
    print bag_fname
    bag = rb.Bag(bag_fname)
    if vals.save_fname == '':
        ed.save_observations_one_camera(bag, calib_fname)
    else:
        ed.save_observations_one_camera(bag, calib_fname, vals.save_fname)