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

# Some global variables
data_dir = os.getenv('HD_DATA_DIR')
map_dir = os.getenv("CAMERA_MAPPING_DIR")

demo_type_dir = None

cmd_checker = None
camera_types = None
downsample = 1

bag_cmd = "rosbag record -O %s /l_pot_angle /r_pot_angle /segment /tf"
kinect_cmd = "record_rgbd_video --out=%s --downsample=%i --device_id=#%i"
webcam_cmd =  "date +%%s.%%N > %s/stamps_init.txt; " + \
"gst-launch -m v4l2src device=/dev/video%i ! video/x-raw-yuv,width=1280,height=960,framerate=30/1 \
! timeoverlay ! ffmpegcolorspace ! jpegenc \
! multifilesink post-messages=true location=\"%s/rgb%%05d.jpg > %s/stamps_info.txt"
voice_cmd = "roslaunch pocketsphinx demo_recording.launch"

# def record_demo (demo_dir, bag_command, camera_commands, use_voice=True)

class voice_alerts ():
    
    def __init__(self):
        
        self.segment_state = None
        self.sub = rospy.Subscriber('/segment', Segment, callback=self.segment_cb)
        
    def segment_cb (self, msg):
        self.segment_state = msg.command
    
    def get_latest_msg (self):
        return self.segment_state

def create_commands_for_demo (demo_dir):
    """
    Creates the command for recording bag files and demos given demo_dir.
    """
    global camera_types
    
    bag_cmd_demo = bag_cmd%(osp.join(demo_dir,'demo'))
    camera_commands = {}
    dev_id = 1
    for cam in camera_types:
        if camera_types[cam] == 'kinect':
            cam_dir = osp.join(demo_dir, 'camera_')
            camera_commands[cam] = kinect_cmd%(cam_dir, downsample, dev_id) 
            dev_id += 1
        else:
            with open(osp.join(map_dir,'camera%i'%cam),'r') as fh: dev_video = int(fh.read())
            webcam_dir = osp.join(demo_dir,'camera_#%i'%(cam))
            try:
                os.mkdir(webcam_dir)
            except:
                print "Directory %d already exists."%webcam_dir

            webcam_cmd_demo = webcam_cmd%(webcam_dir, dev_video, webcam_dir, webcam_dir) 
            camera_commands[cam] = webcam_cmd_demo

    return bag_cmd_demo, camera_commands


def record_demo (master_type, demo_name, calib_file="", num_cameras=2, down_sample=1):
    """
    Record a single demo.
    """
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
            pass
        else:
            if osp.exists(demo_dir):
                print "Removing demo dir" 
                shutil.rmtree(demo_dir)
                print "Done"


def record_pipeline ( demo_type, calib_file = "", 
                        num_camers = 2, num_demos = -1, use_voice = True):
    """
    Either records n demos or until person says he is done.
    @demo_type: type of demo being recorded. specifies directory to save in.
    @calib_file: file to load calibration from.
    @num_cameras: number of cameras being used.
    @num_demos: number of demos to be recorded. -1 -- default -- means until user stops.
    @use_voice: use voice commands if true. o/w use command line.
    
    Note: Need to start up all cameras from ar_track_service before recording
          so that the latest camera type mapping/device mapping is stored and ready.  
    """
    global cmd_checker, camera_types, demo_type_dir

    rospy.init_node("time_to_record")
    sleeper = rospy.Rate(10)

    # Initialize directory and other variables
    demo_type_dir = osp.join(data_dir, demo_type)

    # Taken care of outside.
    master_file = osp.join(demo_type_dir,'master.yaml')
    if not osp.isfile(master_file):
        with open(master_file, "w") as f:
            f.write("name: %s\n"%demo_type)
            f.write("h5path: %s\n"%demo_type+".h5")
            f.write("demos: \n")
    
    camera_types = {(i+1):None for i in range(num_cameras)}
    for cam in camera_types:
        with open(osp.join(data_dir,'camera_types','camera%i'%cam),'r') as fh: camera_types[cam] = fh.read()
    
    # Get number of latest demo recorded
    latest_demo_file = osp.join(demo_type_dir, 'latest_demo.txt')     
    if osp.isfile(latest_demo_file):
        with open(latest_demo_file,'r') as fh:
            demo_num = int(fh.read()) + 1
    else: demo_num = 1
    
    video_dirs = []
    for i in range(1,num_cameras+1):
        video_dirs.append("camera_#%s"%(i))
        
    demo_info = {"- bag_file": "demo.bag",
                 "  video_dirs": video_dirs,
                 "  annotation_file": "ann.yaml",
                 "  data_file": "demo.data",
                 "  traj_file": "demo.traj",
                 "  demo_name": ""}

    # Load calibration
    cpipe.initialize_calibration(args.num_cameras)
    calib_file = osp.join(data_dir, 'calib', calibration_file);
    if osp.isfile(calib_file):
        cpipe.tfm_pub.load_calibration(calib_file)
    else:
        cpipe.run_calibration_sequence()

    # Get voice command and subscriber launched and ready
    greenprint(voice_cmd)
    voice_handle = subprocess.Popen(voice_cmd, shell=True)
    started_voice = True
    cmd_checker = voice_alerts()

    # Start recording demos.
    while True:
        # Check if continuing or stopping
        if use_voice:
            subprocess.call("espeak -v en 'Say begin recording when ready to record.'", shell=True)
            while True:
                status = cmd_checker.get_latest_msg()
                if  status in ["begin recording","done session"]:
                    break
                sleeper.sleep()
        else:
            status = raw_input("Hit enter when ready to record next demo (or q/Q to quit). ")
        
        if status in ["done session", "q","Q"]:
            greenprint("Done recording for this session.")
            break


        # Initialize names and record
        demo_name = "demo%05d"%(demo_num)
        bag_cmd_demo = bag_cmd%(demo_dir+"/demo")
        demo_dir = osp.join(demo_type_dir, demo_name)

        greenprint("Recording %s."%demo_name)
        save_demo = record_demo(demo_dir)
        
        if save_demo:
            with open(master_file, 'a') as fh:
                for item in demo_info:
                    fh.write(item+': '+demo_info[item] + '\n')
            cam_type_file = osp.join(demo_dir, 'camera_types.yaml')
            with open(cam_type_file,"a") as fh: yaml.dump(camera_types)

            with open(latest_demo_file,'w') as fh: fh.write(demo_num)
            demo_num += 1
            
            if num_demos > 0:
                num_demos -= 1
                
            greenprint("Saved %s"%demo_name)
        else:
            if osp.exists(demo_dir):
                print "Removing %s dir"%demo_name 
                shutil.rmtree(demo_dir)
                print "Done"
            
        if num_demos == 0:
            greenprint("Recorded all demos for session.")
            break



if __name__ == '__main__':
    global downsample

    parser = argparse.ArgumentParser()
    parser.add_argument("demo_type", type=str)
    parser.add_argument("demo_name", type=str)
    parser.add_argument("calibration_file", default='')
    parser.add_argument("num_cameras", default=2, type=int)
    parser.add_argument("--downsample", default=1, type=int)
    args = parser.parse_args()

    downsample = args.downsample