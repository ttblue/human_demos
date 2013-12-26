#!/usr/bin/env python

'''
Script to record rgbd data
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
import yaml

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

raw_input("Hit enter when ready to record demo.")
yellowprint("Recording demonstration now...")

try:
    if not osp.exists(demo_dir):
        os.mkdir(demo_dir)
    
    bag_cmd = "rosbag record -O %s /l_pot_angle /r_pot_angle /segment /tf"%(demo_dir+"/demo")
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
    
    started_voice = False    
    voice_cmd = "roslaunch pocketsphinx demo_recording.launch"
    greenprint(voice_cmd)
    voice_handle = subprocess.Popen(voice_cmd, shell=True)
    started_voice = True
    
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
    if started_voice:
        print "stopping voice"
        voice_handle.send_signal(signal.SIGINT)
        voice_handle.wait()
    


    bag_filename = demo_dir+"/demo.bag"
    video1_dirname = demo_dir+"/camera_#1"
    video2_dirname = demo_dir+"/camera_#2"
    annotation_filename = demo_dir+"/ann.yaml"
    
    video_dirs = []
    for i in range(1,args.num_cameras+1):
        video_dirs.append("camera_#%s"%(i))
        
    bag_info = {"bag_file": "demo.bag",
                "video_dirs": video_dirs,
                "annotation_file": "demo.ann.yaml",
                "data_file": "demo.data",
                "traj_file": "demo.traj",
                "demo_name": args.demo_name}
    
    master_info = {"name": args.demo_name,
                   "h5path": args.demo_name + ".h5",
                    "bags": bag_info}
    
    if yes_or_no("save demo?"):
        with open(demo_dir+"/"+args.master_file,"w") as fh:
            yaml.dump(master_info, fh)
    else:
        if osp.exists(demo_dir):
            print "Removing demo dir" 
            shutil.rmtree(demo_dir)
            print "Done"

#exit()