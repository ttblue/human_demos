'''
Script to extract data from rgbd+rosbag demonstration
'''
from hd_track import extract_data as ed
import rosbag as rb
import argparse
import os, os.path as osp


data_dir = os.getenv('HD_DATA_DIR')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-dtype", help="type of demonstration", action='store', dest='demo_type', type=str)
    parser.add_argument("-dname", help="name of demonstration", action='store', dest='demo_name', type=str)
    parser.add_argument("-clib", help="name of calibration file", action='store', dest='calib_fname', type=str)
    parser.add_argument("-cnum", help="number of cameras", action='store', dest="num_cameras", type=int, default=2)
    parser.add_argument("-sname", help="name of save file", action='store', dest='save_fname', default='', type=str)
    vals = parser.parse_args()

    calib_fname = osp.join(data_dir, 'calib', vals.calib_fname)
        
    if vals.save_fname == '':
        ed.save_observations_rgbd(vals.demo_type, vals.demo_name, calib_fname, vals.num_cameras)
    else:
        ed.save_observations_rgbd(vals.demo_type, vals.demo_name, calib_fname, vals.num_cameras, vals.save_fname)