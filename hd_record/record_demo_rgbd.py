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

from generate_webcam_timestamps import gen_timestamps
from hd_utils.defaults import demo_files_dir, calib_files_dir, data_dir

devnull = open(os.devnull, 'wb')
# Some global variables
map_dir = os.getenv("CAMERA_MAPPING_DIR")

demo_type_dir = None
latest_demo_file = None
cmd_checker = None
camera_types = None
demo_info = None
prefix = None
demo_num = 0

bag_cmd = "rosbag record -O %s /l_pot_angle /r_pot_angle /segment /tf"
kinect_cmd = "record_rgbd_video --out=%s --downsample=%i --device_id=#%i"
webcam_cmd =  "date +%%s.%%N > %s/stamps_init.txt; " + \
"gst-launch -m v4l2src device=/dev/video%i ! video/x-raw-yuv,width=1280,framerate=30/1 \
! ffmpegcolorspace ! jpegenc \
! multifilesink post-messages=true location=\"%s/rgb%%05d.jpg\" > %s/stamps_info.txt"
voice_cmd = "roslaunch pocketsphinx demo_recording.launch"

def terminate_process_and_children(p):
    ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % p.pid, shell=True, stdout=subprocess.PIPE)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()
    assert retcode == 0, "ps command returned %d" % retcode
    for pid_str in ps_output.split("\n")[:-1]:
            os.kill(int(pid_str), signal.SIGINT)
    p.terminate()

class voice_alerts ():
    """
    Class to process and use voice commands.
    """
    def __init__(self):
        
        self.segment_state = None
        self.sub = rospy.Subscriber('/segment', Segment, callback=self.segment_cb)
        
    def segment_cb (self, msg):
        self.segment_state = msg.command
    
    def get_latest_msg (self):
        return self.segment_state

def load_parameters (demo_type, num_cameras):
    """
    Initialize global variables.
    """
    global camera_types, demo_type_dir, master_file, demo_num, demo_info, latest_demo_file, prefix
    demo_type_dir = osp.join(demo_files_dir, demo_type)
    if not osp.isdir(demo_type_dir):
        os.mkdir(demo_type_dir)

    # Taken care of outside.
    master_file = osp.join(demo_type_dir,'master.yaml')
    if not osp.isfile(master_file):
        with open(master_file, "w") as f:
            f.write("name: %s\n"%demo_type)
            f.write("h5path: %s\n"%(demo_type+".h5"))
            f.write("demos: \n")

    camera_types = {(i+1):None for i in range(num_cameras)}
    for cam in camera_types:
        with open(osp.join(data_dir,'camera_types','camera%i'%cam),'r') as fh: camera_types[cam] = fh.read()
    
    # Get number of latest demo recorded
    latest_demo_file = osp.join(demo_type_dir, 'latest_demo.txt')     
    if osp.isfile(latest_demo_file):
        try:
            with open(latest_demo_file,'r') as fh:
                demo_num = int(fh.read()) + 1
        except: demo_num = 1
    else: demo_num = 1
                                                                                                                                                        
    video_dirs = []
    for i in range(1,num_cameras+1):
        video_dirs.append("camera_#%s"%(i))
        
    demo_info = {"bag_file": "demo.bag",
                 "video_dirs": str(video_dirs),
                 "annotation_file": "ann.yaml",
                 "data_file": "demo.data",
                 "traj_file": "demo.traj",
                 "demo_name": ""}
    prefix = ["  " for _ in demo_info]
    prefix[0] = "- "

def create_commands_for_demo (demo_dir):
    """
    Creates the command for recording bag files and demos given demo_dir.
    """
    global camera_types
    
    bag_cmd_demo = bag_cmd%(osp.join(demo_dir,'demo'))
    camera_commands = {}
    dev_id = 1
    for cam in camera_types:
        if camera_types[cam] == 'rgbd':
            cam_dir = osp.join(demo_dir, 'camera_')
            camera_commands[cam] = kinect_cmd%(cam_dir, downsample, dev_id) 
            dev_id += 1
        else:
            with open(osp.join(map_dir,'camera%i'%cam),'r') as fh: dev_video = int(fh.read())
            webcam_dir = osp.join(demo_dir,'camera_#%i'%(cam))
            try:
                os.mkdir(webcam_dir)
            except:
                print "Directory %s already exists."%webcam_dir

            webcam_cmd_demo = webcam_cmd%(webcam_dir, dev_video, webcam_dir, webcam_dir) 
            camera_commands[cam] = webcam_cmd_demo

    return bag_cmd_demo, camera_commands


def record_demo (bag_cmd_demo, camera_commands, use_voice=True):
    """
    Record a single demo.
    Returns True/False depending on whether demo needs to be saved.
    
    @bag_command and @camera_commands are the commands to be called for recording.
    """
    global cmd_checker
    # Start here. Change to voice command.

    sleeper = rospy.Rate(10)
    try:
        started_bag = False
        started_video = {}
        video_handles= {}

        greenprint(bag_cmd_demo)
        bag_handle = subprocess.Popen(bag_cmd_demo, stdout=devnull, stderr=devnull, shell=True)
        time.sleep(1)
        poll_result = bag_handle.poll() 
        if poll_result is not None:
            print "poll result", poll_result
            raise Exception("problem starting bag recording")
        started_bag = True

        for cam in camera_commands:
            greenprint(camera_commands[cam])
            video_handles[cam] = subprocess.Popen(camera_commands[cam], stdout=devnull, stderr=devnull, shell=True)
            started_video[cam] = True
        

        # Change to voice command
        if use_voice:
            subprocess.call("espeak -v en 'Recording.'", stdout=devnull, stderr=devnull, shell=True)
            while True:
                status = cmd_checker.get_latest_msg()
                if  status in ["cancel recording","finish recording"]:
                    break
                sleeper.sleep()
        else:
            yellowprint("Press ctrl-c when done. ")
            time.sleep(9999)

    except Exception as e:
        greenprint(e)#"got control-c")
        if use_voice: status == "cancel_recording"
    
    finally:
        cpipe.done()
        if started_bag:
            yellowprint("stopping bag")
            terminate_process_and_children(bag_handle)
            #bag_handle.send_signal(signal.SIGINT)
            bag_handle.wait()
            yellowprint("stopped bag")
        for cam in started_video:
            if started_video[cam]:
                yellowprint("stopping  video%i"%cam)
                terminate_process_and_children(video_handles[cam])
                #video_handles[cam].send_signal(signal.SIGINT)
                video_handles[cam].wait()
                yellowprint("stopped  video%i"%cam)
        
        if use_voice:
            return status == "finish recording"
        elif yes_or_no("save demo?"):
            return True
        else:
            return False


def record_pipeline ( demo_type, calib_file = "", 
                        num_cameras = 2, num_demos = -1, use_voice = True):
    """
    Either records n demos or until person says he is done.
    @demo_type: type of demo being recorded. specifies directory to save in.
    @calib_file: file to load calibration from.
    @num_cameras: number of cameras being used.
    @num_demos: number of demos to be recorded. -1 -- default -- means until user stops.
    @use_voice: use voice commands to start/stop demo if true. o/w use command line.
    
    Note: Need to start up all cameras from ar_track_service before recording
          so that the latest camera type mapping/device mapping is stored and ready.  
    """
    global cmd_checker, camera_types, demo_type_dir, master_file, demo_num, demo_info, latest_demo_file

    load_parameters(demo_type, num_cameras)

    rospy.init_node("time_to_record")
    sleeper = rospy.Rate(10)

    # Load calibration
    cpipe.initialize_calibration(args.num_cameras)
    calib_file = osp.join(calib_files_dir, calib_file);
    if osp.isfile(calib_file):
        cpipe.tfm_pub.load_calibration(calib_file)
    else:
        cpipe.run_calibration_sequence()

    # Get voice command and subscriber launched and ready
    greenprint(voice_cmd)
    voice_handle = subprocess.Popen(voice_cmd, stdout=devnull, stderr=devnull, shell=True)
    started_voice = True
    cmd_checker = voice_alerts()

    # Start recording demos.
    while True:
        # Check if continuing or stopping
        if use_voice:
            time.sleep(1.2)
            subprocess.call("espeak -v en 'Ready.'", stdout=devnull, stderr=devnull, shell=True)
            while True:
                status = cmd_checker.get_latest_msg()
                if  status in ["begin recording","done session"]:
                    break
                sleeper.sleep()
        else:
            status = raw_input("Hit enter when ready to record demo (or q/Q to quit). ")

        if status in ["done session", "q","Q"]:
            greenprint("Done recording for this session.")
            exit()


        # Initialize names and record
        demo_name = "demo%05d"%(demo_num)        
        demo_dir = osp.join(demo_type_dir, demo_name)
        if not osp.exists(demo_dir): os.mkdir(demo_dir)

        bag_cmd, camera_commands = create_commands_for_demo (demo_dir)

        greenprint("Recording %s."%demo_name)
        save_demo = record_demo(bag_cmd, camera_commands, use_voice)
        
        if save_demo:
            demo_info["demo name"] = demo_name
            with open(master_file, 'a') as fh:
                for i, item in enumerate(demo_info):
                    fh.write(prefix[i]+item+': '+demo_info[item] + '\n')

            cam_type_file = osp.join(demo_dir, 'camera_types.yaml')
            with open(cam_type_file,"w") as fh: yaml.dump(camera_types, fh)
            
            for cam in camera_types:
                if camera_types[cam] == "rgb":
                    gen_timestamps(osp.join(demo_dir, 'camera_#%i'%cam))

            with open(latest_demo_file,'w') as fh: fh.write(str(demo_num))
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

def record_single_demo (demo_type, demo_name, calib_file = "", 
                          num_cameras = 2, use_voice = True):
    """
    Records a single demo.
    @demo_type: type of demo being recorded. specifies directory to save in.
    @demo_name: name of demo.
    @calib_file: file to load calibration from.
    @num_demos: number of demos to be recorded. -1 -- default -- means until user stops.
    @use_voice: use voice commands to start/stop demo if true. o/w use command line.
    
    Note: Need to start up all cameras from ar_track_service before recording
          so that the latest camera type mapping/device mapping is stored and ready.  
    """
    global cmd_checker, camera_types, demo_type_dir, master_file, demo_info

    load_parameters(demo_type, num_cameras)

    rospy.init_node("time_to_record")
    sleeper = rospy.Rate(10)

    # Load calibration
    cpipe.initialize_calibration(args.num_cameras)
    calib_file = osp.join(calib_files_dir, calibration_file);
    if osp.isfile(calib_file):
        cpipe.tfm_pub.load_calibration(calib_file)
    else:
        cpipe.run_calibration_sequence()

    # Get voice command and subscriber launched and ready
    greenprint(voice_cmd)
    voice_handle = subprocess.Popen(voice_cmd, stdout=devnull, stderr=devnull, shell=True)
    started_voice = True
    cmd_checker = voice_alerts()

    # Check if continuing or stopping
    if use_voice:

        subprocess.call("espeak -v en 'Ready.'", stdout=devnull, stderr=devnull, shell=True)
        while True:
            status = cmd_checker.get_latest_msg()
            if  status in ["begin recording","done session"]:
                break
            sleeper.sleep()
    else:
        status = raw_input("Hit enter when ready to record demo (or q/Q to quit). ")
    
    if status in ["done session", "q","Q"]:
        greenprint("Done recording for this session (already?).")
        return

    # Initialize names and record
    demo_dir = osp.join(demo_type_dir, demo_name)
    if not osp.exists(demo_dir): os.mkdir(demo_dir)

    bag_cmd, camera_commands = create_commands_for_demo (demo_dir)

    greenprint("Recording %s."%demo_name)
    save_demo = record_demo(bag_cmd, camera_commands, use_voice)

    if save_demo:
        demo_info["demo name"] = demo_name
        with open(master_file, 'a') as fh:
            for i, item in enumerate(demo_info):
                fh.write(prefix[i]+item+': '+demo_info[item] + '\n')
                
        for cam in camera_types:
            if camera_types[cam] == "rgb":
                gen_timestamps(osp.join(demo_dir, 'camera_#%i'%cam))
        cam_type_file = osp.join(demo_dir, 'camera_types.yaml')
        with open(cam_type_file,"w") as fh: yaml.dump(camera_types, fh)
        
        greenprint("Saved %s"%demo_name)
    else:
        if osp.exists(demo_dir):
            print "Removing demo %s"%demo_name 
            shutil.rmtree(demo_dir)
            print "Done"

if __name__ == '__main__':
    global downsample

    parser = argparse.ArgumentParser()
    parser.add_argument("demo_type", help="Type of demonstration.", type=str)
    
    parser.add_argument("calibration_file", help="Calibration file.", default='')
    parser.add_argument("num_cameras", help="Number of cameras in setup.", default=2, type=int)
    parser.add_argument("--downsample", help="Downsample rgbd data by factor.", default=1, type=int)
    parser.add_argument("--use_voice", help="Use voice for recording.", default=True, type=bool)
    
    parser.add_argument("--single_demo", help="Single or multiple demos?", action="store_true", default=False)
    parser.add_argument("--num_demos", help="Number of demos to be recorded.", default=-1, type=int)
    parser.add_argument("--demo_name", help="Name of demo if single demo.", default="", type=str)
    args = parser.parse_args()

    downsample = args.downsample
    
    if args.single_demo:
        record_single_demo (demo_type = args.demo_type, demo_name=args.demo_name,
                            calib_file = args.calibration_file, num_cameras = args.num_cameras, 
                            use_voice = args.use_voice)
    else:
        record_pipeline (demo_type = args.demo_type, calib_file = args.calibration_file, 
                         num_cameras = args.num_cameras, num_demos = args.num_demos, 
                         use_voice = args.use_voice)