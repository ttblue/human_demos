#!/usr/bin/env python

'''
Script to record demos with rgbd data.
'''

import argparse
import subprocess, signal
import os, os.path as osp
import itertools
import rospy, roslib
import time, os, shutil
import yaml
import threading

roslib.load_manifest('pocketsphinx')
from pocketsphinx.msg import Segment

from hd_calib import calibration_pipeline as cpipe
from hd_utils.colorize import *
from hd_utils.yes_or_no import yes_or_no

bag_cmd = "rosbag record -O %s /l_pot_angle /r_pot_angle /segment /tf"
voice_cmd = "roslaunch pocketsphinx demo_recording.launch"

# def record_demo (demo_dir, bag_command, camera_commands, use_voice=True)

class voice_alerts ():
    
    def __init__(self):
        
        self.segment_state = None
        self.sub = rospy.Subscriber('/segment', Segment, callback=self.segment_cb)
        
    def segment_cb (self, msg):
        self.segment_state = msg.command
    
    def get_latest_msg (self):


def record_demo (master_type, demo_name, calib_file="", num_cameras=2, down_sample=1):
    
    # Start here. Change to voice command.
    raw_input("Hit enter when ready to record demo.")
    yellowprint("Recording demonstration now...")
    
    try:
        # Outside
        if not osp.exists(demo_dir):
            os.mkdir(demo_dir)
        
        started_bag = False
        started_video = {}
        video_handles= {}

        greenprint(bag_cmd)
        bag_handle = subprocess.Popen(bag_cmd, shell=True)
        time.sleep(1)
        poll_result = bag_handle.poll() 
        if poll_result is not None:
            print "poll result", poll_result
            raise Exception("problem starting bag recording")
        else: started_bag = True
    
        for cam in camera_commands:
            greenprint(camera_commands[cam])
            video_handle[cam] = subprocess.Popen(camera_commands[cam], shell=True)
            started_video[cam] = True
        
        # Outside
        greenprint(voice_cmd)
        voice_handle = subprocess.Popen(voice_cmd, shell=True)
        started_voice = True
        
        # Change to voice command
        time.sleep(9999)    
    
    except KeyboardInterrupt:
        greenprint("got control-c")
    
    finally:
        cpipe.done()
        if started_bag:
            print "stopping bag"
            bag_handle.send_signal(signal.SIGINT)
            bag_handle.wait()
        for cam in started_video:
            if started_video[cam]:
                print "stopping video%i"%cam
                video_handle[cam].send_signal(signal.SIGINT)
                video_handle[cam].wait()
    
    
        # All this can be done outside.
        bag_filename = demo_dir+"/demo.bag"
        annotation_filename = demo_dir+"/ann.yaml"
        
        video_dirs = []
        for i in range(1,args.num_cameras+1):
            video_dirs.append("camera_#%s"%(i))
            
        demo_info = {"- bag_file": "demo.bag",
                     "  video_dirs": video_dirs,
                     "  annotation_file": "ann.yaml",
                     "  data_file": "demo.data",
                     "  traj_file": "demo.traj",
                     "  demo_name": args.demo_name}
        
        master_info = {"name": args.demo_name,
                       "h5path": args.demo_name + ".h5",
                       "bags": demo_info}
        if yes_or_no("save demo?"):
            with open(master_file,"a") as fh:
                for item in demo_info:
                    fh.write(item+': '+demo_info[item] + '\n')
            cam_type_file = osp.join(demo_dir, 'camera_types.yaml')
            with open(cam_type_file,"a") as fh: yaml.dump(camera_types)
        else:
            if osp.exists(demo_dir):
                print "Removing demo dir" 
                shutil.rmtree(demo_dir)
                print "Done"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_type", type=str)
    parser.add_argument("demo_name", type=str)
    parser.add_argument("calibration_file", default='')
    parser.add_argument("num_cameras", default=2, type=int)
    parser.add_argument("--downsample", default=1, type=int)
    args = parser.parse_args()

    # Taken care of outside.
    data_dir = os.getenv('HD_DATA_DIR')
    master_file = osp.join(data_dir, args.demo_type, 'master.yaml')
    if not osp.isfile(master_file):
        with open(master_file, "w") as f:
            f.write("name: %s\n"%args.demo_type)
            f.write("h5path: %s\n"%(args.demo_type+".h5"))
            f.write("demos: \n")
    # This too.
    if args.calibration_file == '':
        yellowprint("need calibration file")
        return false
    
    # This too.
    calib_file = osp.join(data_dir, 'calib', args.calibration_file);
    cpipe.initialize_calibration(args.num_cameras)
    cpipe.tfm_pub.load_calibration(calib_file)
    
    #This too.
    demo_dir = osp.join(data_dir, 'demos', args.demo_type, args.demo_name)

    # Outside
    bag_cmd_demo = bag_cmd%(demo_dir+"/demo")
    kinect_cmd = "record_rgbd_video --out=%s --downsample=%i"%(demo_dir+"/camera_", args.downsample) + "--device_id=#%i"
    webcam_cmd = \
        "gst-launch -m v4l2src device=/dev/video%i ! video/x-raw-yuv,width=1280,height=960,framerate=30/1 \
    ! timeoverlay ! ffmpegcolorspace ! jpegenc \
    ! multifilesink post-messages=true location=\"%s/rgb%%05d.jpg"
    
    #Outside
    camera_types = {(i+1):None for i in range(args.num_cameras)}
    for cam in camera_types:
        with open(osp.join(demo_dir,'camera%i'%cam),'r') as fh: camera_types[cam] = fh.read()
    # Outside
    camera_commands = {}
    map_dir = os.getenv("CAMERA_MAPPING_DIR")
    dev_id = 1
    for cam in camera_types:
        if camera_types[cam] == 'kinect':
            camera_commands[cam] = kinect_cmd%dev_id 
            dev_id += 1
        else:
            with open(osp.join(map_dir,'camera%i'%cam),'r') as fh: dev_video = int(fh.read())
            webcam_dir = osp.join(demo_dir,'camera_#%i'%(cam))
            try:
                os.mkdir(webcam_dir)
            except:
                print "Directory %d already exists."%webcam_dir
            ts_command = "date +%s.%N > " + webcam_dir + "stamps_init.txt; " 
            save_command = webcam_cmd%(dev_video,webcam_dir) + " > %s"%osp.join(webcam_dir,"stamps_info.txt") 
            camera_commands[cam] = ts_command + save_command
