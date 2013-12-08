#!/usr/bin/env python

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
rgbd_dir = osp.join(data_dir, 'rgdb', args.demo_name)


raw_input("Hit enter when ready to record demo.")
yellowprint("Recording demonstration now...")

try:
    bag_cmd = "rosbag record -O %s /pot_angle /tf"%(demo_dir)
    greenprint(bag_cmd)
    bag_handle = subprocess.Popen(bag_cmd, shell=True)
    time.sleep(1)
    poll_result = bag_handle.poll() 
    
    if poll_result is not None:
        print "poll result", poll_result
        raise Exception("problem starting bag recording")
    else: started_bag = True
    
    
    video_cmd1 = "record_rgbd_video --out=%s --downsample=%i"%(demo_dir+"1", args.downsample)
    greenprint(video_cmd1)
    if args.num_cameras == 2:
        video_cmd2 = "record_rgbd_video --out=%s --downsample=%i --device_id=#2"%(demo_dir+"2", args.downsample)
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


# bagfilename = demo_dir+".bag"
# if yes_or_no("save demo?"):
#     with open(args.master_file,"w") as fh:
#         fh.write("\n"
#             "- bag_file: %(bagfilename)s\n"
#             "  video_dir: %(videodir)s\n"
#             "  demo_name: %(demoname)s"%dict(bagfilename=bagfilename, videodir=demo_dir, demoname=demo_dir))
# else:
#     if osp.exists(demo_dir):
#         print "Removing demo dir" 
#         shutil.rmtree(demo_dir)
#         print "Done"
#     if osp.exists(bagfilename):
#         print "Removing bag file"
#         os.unlink(bagfilename)
#         print "Done"

#exit()