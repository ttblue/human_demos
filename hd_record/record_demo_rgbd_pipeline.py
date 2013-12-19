#!/usr/bin/env python

'''
The entire pipeline to record rgbd data

Does not work....

'''

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("master_file", type=str)
parser.add_argument("demo_name", type=str)
parser.add_argument("calibration_file", default='')
parser.add_argument("num_cameras", default=2, type=int)
parser.add_argument("--downsample", default=1, type=int)
args = parser.parse_args()

import subprocess, signal
import os, os.path as osp
import itertools
import rospy
import time, os, shutil

from hd_calib import calibration_pipeline as cpipe
from hd_utils.colorize import *
from hd_utils.yes_or_no import yes_or_no

from hd_track import extract_data as ed
import rosbag as rb
from hd_track.tracking_utility import *
import cPickle

from hd_utils.defaults import demo_files_dir, calib_files_dir


started_bag = False
started_video1 = False
started_video2 = False


data_dir = os.getenv('HD_DATA_DIR')
master_file = osp.join(data_dir, 'demos', args.master_file)


if args.calibration_file == '':
    yellowprint("need calibration file")
    
    
calib_file = osp.join(data_dir, 'calib', args.calibration_file);
cpipe.initialize_calibration(args.num_cameras)
cpipe.tfm_pub.load_calibration(calib_file)

demo_dir = osp.join(data_dir, 'demos', args.demo_name)

raw_input("Hit enter when ready to record demo.")
yellowprint("Recording demonstration now...")

try:
    if not osp.exists(demo_dir):
        os.mkdir(demo_dir)
    
    bag_cmd = "rosbag record -O %s /l_pot_angle /r_pot_angle /tf"%(demo_dir+"/demo.bag")
    greenprint(bag_cmd)
    bag_handle = subprocess.Popen(bag_cmd, shell=True)
    time.sleep(1)
    poll_result = bag_handle.poll() 
    
    if poll_result is not None:
        print "poll result", poll_result
        raise Exception("problem starting bag recording")
    else: started_bag = True
    
  
    video_cmd1 = "record_rgbd_video --out=%s --downsample=%i"%(demo_dir+"/camera_", args.downsample)
    greenprint(video_cmd1)
    if args.num_cameras == 2:
        video_cmd2 = "record_rgbd_video --out=%s --downsample=%i --device_id=#2"%(demo_dir+"/camera_", args.downsample)
        greenprint(video_cmd2)

    started_video2 = False
    if args.num_cameras == 2:
        video_handle1 = subprocess.Popen(video_cmd1, shell=True)
        video_handle2 = subprocess.Popen(video_cmd2, shell=True)
        started_video1 = True
        started_video2 = True
    else:
        video_handle1 = subprocess.Popen(video_cmd1, shell=True)
        started_video1 = True    

    time.sleep(9999)    

except KeyboardInterrupt:
    greenprint("got control-c")

finally:
    cpipe.done()
    if started_bag:
        print "stopping bag"
        bag_handle.send_signal(signal.SIGINT)
        bag_handle.wait()
    if started_video1:
        print "stopping video1"
        video_handle1.send_signal(signal.SIGINT)
        video_handle1.wait()
    if started_video2:
        print "stopping video2"
        video_handle2.send_signal(signal.SIGINT)
        video_handle2.wait()

time.sleep(100)    

bag_filename = osp.join(demo_dir, 'demo.bag')
video1_dirname = osp.join(demo_dir, 'camera_#1')  
video2_dirname = osp.join(demo_dir, 'camera_#2')
annotation_filename = osp.join(demo_dir, 'ann.yaml')
data_filename = osp.join(demo_dir, "demo.data")
traj_filename = osp.join(demo_dir, "demo.traj")

ed.save_observations_rgbd(args.demo_name, args.calibration_file)


freq = 30
demo_fname = vals.demo_fname
calib_fname = vals.calib_fname

traj_data = traj_kalman(data_filename, calib_file, freq)
    
with open(traj_filename, 'w') as fh:
    cPickle.dump(traj_data, fh)


if yes_or_no("save demo?"):
    with open(demo_dir+"/"+args.master_file,"w") as fh:
        fh.write("\n"
            "- bag_file: %(bagfilename)s\n"
            "  video_dir1: %(video1dir)s\n"
            "  video_dir2: %(video2dir)s\n"
            "  annot_dir: %(annofilename)s\n"
            "  data_file: %(datafilename)s\n"
            "  traj_file: %(trajfilename)s\n"
            "  demo_name: %(demoname)s"
            %dict(bagfilename=bag_filename, video1dir=video1_dirname, 
                  video2dir=video2_dirname, annofilename=annotation_filename,
                  datafilename=data_filename, trajfilename=traj_filename,
                  demoname=demo_dir))
else:
    if osp.exists(demo_dir):
        print "Removing demo dir" 
        shutil.rmtree(demo_dir)
        print "Done"

#exit()