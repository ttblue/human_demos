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
from std_msgs.msg import Float32
roslib.load_manifest('tf')
from tf.msg import tfMessage

roslib.load_manifest('record_rgbd_service')
from record_rgbd_service.srv import SaveImage, SaveImageRequest, SaveImageResponse

from hd_calib import calibration_pipeline as cpipe
from hd_utils.colorize import *
from hd_utils.yes_or_no import yes_or_no

from hd_utils.defaults import demo_files_dir, calib_files_dir, data_dir, \
                              demo_names, master_name, latest_demo_name
from hd_utils.utils import terminate_process_and_children
from hd_utils.yes_or_no import yes_or_no

from generate_annotations import generate_annotation
from rosbag_service import TopicWriter

devnull = open(os.devnull, 'wb')
# Some global variables
map_dir = os.getenv("CAMERA_MAPPING_DIR")

topic_writer = None

save_image_services = {}
cam_save_requests = {}
cam_stop_request = None

demo_type_dir = None
latest_demo_file = None
cmd_checker = None
camera_types = None
camera_models = {}
demo_num = 0

voice_cmd = "roslaunch pocketsphinx demo_recording.launch"

class voice_alerts ():
    """
    Class to process and use voice commands.
    """
    def __init__(self):
        
        self.segment_state = None
        self.sub = rospy.Subscriber('/segment', Segment, callback=self.segment_cb)
        
    def segment_cb (self, msg):
        self.segment_state = msg.command
        blueprint(self.segment_state)
    
    def get_latest_msg (self):
        return self.segment_state

def load_parameters (demo_type, num_cameras):
    """
    Initialize global variables.
    """
    global camera_types, demo_type_dir, master_file, demo_num, demo_info,\
           latest_demo_file, cam_stop_request, topic_writer

    demo_type_dir = osp.join(demo_files_dir, demo_type)
    if not osp.isdir(demo_type_dir):
        os.mkdir(demo_type_dir)

    # Taken care of outside.
    master_file = osp.join(demo_type_dir, master_name)
    if not osp.isfile(master_file):
        with open(master_file, "w") as f:
            f.write("name: %s\n"%demo_type)
            f.write("h5path: %s\n"%(demo_type+".h5"))
            f.write("demos: \n")

    camera_types = {(i+1):None for i in range(num_cameras)}
    for cam in camera_types:
        with open(osp.join(data_dir,'camera_types','camera%i'%cam),'r') as fh: camera_types[cam] = fh.read()
        save_image_services[cam] = rospy.ServiceProxy("saveImagescamera%i"%cam, SaveImage)
        cam_save_requests[cam] = SaveImageRequest()
        cam_save_requests[cam].start = True 

        if camera_types[cam] == "rgb":
            with open(osp.join(data_dir,'camera_types','camera%i_model'%cam),'r') as fh: camera_models[cam] = fh.read()
            
    cam_stop_request = SaveImageRequest()
    cam_stop_request.start = False
            
    # Get number of latest demo recorded
    latest_demo_file = osp.join(demo_type_dir, latest_demo_name)
    if osp.isfile(latest_demo_file):
        try:
            with open(latest_demo_file,'r') as fh:
                demo_num = int(fh.read()) + 1
        except: demo_num = 1
    else: demo_num = 1
    
    topics = ['/l_pot_angle', '/r_pot_angle','/segment','/tf']
    topic_types = [Float32, Float32, Segment, tfMessage]
    topic_writer = TopicWriter(topics=topics, topic_types=topic_types)
    yellowprint("Started listening to the different topics.")


def stop_camera_saving ():
    for cam in camera_types:
        try:
            save_image_services[cam](cam_stop_request)
        except:
            redprint("Something wrong with camera%i."%cam)
    greenprint("Stopped cameras")
    


def ready_service_for_demo (demo_dir):
    """
    Creates the command for recording bag files and demos given demo_dir.
    """
    global camera_types, camera_save_requests
    
    for cam in camera_types:
        cam_dir = osp.join(demo_dir,'camera_#%i'%(cam))
        if osp.isdir(cam_dir):
            shutil.rmtree(cam_dir)
        os.mkdir(cam_dir)
        cam_save_requests[cam].folder_name = demo_dir


def record_demo (demo_dir, use_voice):
    """
    Record a single demo.
    Returns True/False depending on whether demo needs to be saved.
    
    @demo_dir: directory where demo is recorded.
    @use_voice: bool on whether to use voice for demo or not.
    """
    global cmd_checker, topic_writer
    # Start here. Change to voice command.

    sleeper = rospy.Rate(30)

    started_bag = False
    started_video = {cam:False for cam in camera_types}

    print
    yellowprint("Starting bag file recording.")
    bag_file = osp.join(demo_dir,demo_names.bag_name)
    topic_writer.start_saving(bag_file)
    started_bag = True

    for cam in camera_types:
        yellowprint("Calling saveImagecamera%i service."%cam)
        save_image_services[cam](cam_save_requests[cam])
        started_video[cam] = True
    
    # Change to voice command
    time.sleep(1.2)
    subprocess.call("espeak -v en 'Recording.'", stdout=devnull, stderr=devnull, shell=True)
    time_start = time.time()
    if use_voice:
        while True:
            status = cmd_checker.get_latest_msg()
            if  status in ["cancel recording","finish recording","stop recording"]:
                break
            sleeper.sleep()
    else:
        raw_input(colorize("Press any key when done.",'y',True))
    
    time_finish = time.time()
    
    for cam in started_video:
        if started_video[cam]:
            save_image_services[cam](cam_stop_request)
            yellowprint("Stopped video%i."%cam)
    if started_bag:
        topic_writer.stop_saving()
        yellowprint("Stopped bag.")
    
    
    if use_voice:
        while status == "stop recording":
            sleeper.sleep()
            status = cmd_checker.get_latest_msg()
        greenprint("Time taken to record demo: %02f s"%(time_finish-time_start))
        return status == "finish recording"
    elif yes_or_no("Save demo?"):
        greenprint("Time taken to record demo: %02f s"%(time_finish-time_start))
        return True
    else:
        return False


def record_pipeline ( demo_type, calib_file, 
                        num_cameras, num_demos, use_voice):
    """
    Either records n demos or until person says he is done.
    @demo_type: type of demo being recorded. specifies directory to save in.
    @calib_file: file to load calibration from. "", -- default -- means using the one in the master file
    @num_cameras: number of cameras being used.
    @num_demos: number of demos to be recorded. -1 -- default -- means until user stops.
    @use_voice: use voice commands to start/stop demo if true. o/w use command line.
    """
    global cmd_checker, camera_types, demo_type_dir, master_file, demo_num, latest_demo_file, topic_writer

    time_sess_start = time.time()

    rospy.init_node("time_to_record")
    sleeper = rospy.Rate(10)
    
    load_parameters(demo_type, num_cameras)

    # Load calibration
    cpipe.initialize_calibration(args.num_cameras)
    calib_file_path = osp.join(calib_files_dir, calib_file);
    if osp.isfile(calib_file_path):
        cpipe.tfm_pub.load_calibration(calib_file_path)
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
        print '\n\n'
        time.sleep(1.2)
        greenprint("Ready.")
        subprocess.call("espeak -v en 'Ready.'", stdout=devnull, stderr=devnull, shell=True)
        if use_voice:
            time.sleep(1.2)
            while True:
                status = cmd_checker.get_latest_msg()
                if  status in ["begin recording","done session"]:
                    break
                sleeper.sleep()
        else:
            status = raw_input("Hit enter when ready to record demo (or q/Q to quit). ")

        if status in ["done session", "q","Q"]:
            greenprint("Done recording for this session.")
            break


        # Initialize names and record
        demo_name = demo_names.base_name%(demo_num)
        demo_dir = osp.join(demo_type_dir, demo_name)
        if not osp.exists(demo_dir): os.mkdir(demo_dir)
        else:
            yellowprint("%s exists! Removing directory for fresh recording."%s)
            shutil.rmtree(demo_dir)
            os.mkdir(demo_dir)
            

        ready_service_for_demo (demo_dir)

        greenprint("Recording %s."%demo_name)
        save_demo = record_demo(demo_dir, use_voice)
        
        if save_demo:
            with open(master_file, 'a') as fh: fh.write('- demo_name: %s\n'%demo_name)
            
            cam_type_file = osp.join(demo_dir, demo_names.camera_types_name)
            with open(cam_type_file,"w") as fh: yaml.dump(camera_types, fh)
            cam_model_file = osp.join(demo_dir, demo_names.camera_models_name)
            with open(cam_model_file,"w") as fh: yaml.dump(camera_models, fh)
            
            with open(latest_demo_file,'w') as fh: fh.write(str(demo_num))
            demo_num += 1
            
            shutil.copyfile(calib_file_path, osp.join(demo_dir,demo_names.calib_name))
            
            generate_annotation(demo_type, demo_name)
            
            if num_demos > 0:
                num_demos -= 1
                
            greenprint("Saved %s."%demo_name)
        else:
            if osp.exists(demo_dir):
                shutil.rmtree(demo_dir)
                yellowprint("Removed %s dir."%demo_name)
        if num_demos == 0:
            greenprint("Recorded all demos for session.")
            break
        
    if started_voice:
        terminate_process_and_children(voice_handle)
        voice_handle.wait()
        yellowprint("Stopped voice.")
    
    stop_camera_saving()    
    cpipe.done() 
    topic_writer.done_session()
    
    time_sess_finish = time.time()
    greenprint("Time taken to record in this session: %02f s"%(time_sess_finish-time_sess_start))

def record_single_demo (demo_type, demo_name, calib_file, 
                          num_cameras, use_voice):
    """
    Records a single demo.
    @demo_type: type of demo being recorded. specifies directory to save in.
    @demo_name: name of demo.
    @calib_file: file to load calibration from. "", -- default -- means using the one in the master file
    @num_demos: number of demos to be recorded. -1 -- default -- means until user stops.
    @use_voice: use voice commands to start/stop demo if true. o/w use command line.   
    """
    global cmd_checker, camera_types, demo_type_dir, master_file, topic_writer

    rospy.init_node("time_to_record")
    sleeper = rospy.Rate(10)
    
    load_parameters(demo_type, num_cameras)

    # Load calibration
    cpipe.initialize_calibration(args.num_cameras)
    calib_file_path = osp.join(calib_files_dir, calib_file);
    if osp.isfile(calib_file_path):
        cpipe.tfm_pub.load_calibration(calib_file_path)
    else:
        cpipe.run_calibration_sequence()

    # Get voice command and subscriber launched and ready
    greenprint(voice_cmd)
    voice_handle = subprocess.Popen(voice_cmd, stdout=devnull, stderr=devnull, shell=True)
    started_voice = True
    cmd_checker = voice_alerts()

    # Check if continuing or stopping
    greenprint("Ready.")
    subprocess.call("espeak -v en 'Ready.'", stdout=devnull, stderr=devnull, shell=True)
    if use_voice:
        time.sleep(1.2)        
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
    else:
        shutil.rmtree(demo_dir)
        os.mkdir(demo_dir)

    ready_service_for_demo (demo_dir)

    greenprint("Recording %s."%demo_name)
    save_demo = record_demo(demo_dir, use_voice)

    if save_demo:
        with open(master_file, 'a') as fh: fh.write('- demo_name: %s\n'%demo_name)
        
        cam_type_file = osp.join(demo_dir, demo_names.camera_types_name)
        with open(cam_type_file,"w") as fh: yaml.dump(camera_types, fh)
        cam_model_file = osp.join(demo_dir, demo_names.camera_models_name)
        with open(cam_model_file,"w") as fh: yaml.dump(camera_models, fh)

        shutil.copyfile(calib_file_path, osp.join(demo_dir,demo_names.calib_name))
        
        generate_annotation(demo_type, demo_name)
        
        greenprint("Saved %s."%demo_name)
    else:
        time.sleep(3)
        if osp.exists(demo_dir):
            yellowprint("Removing demo %s"%demo_name) 
            shutil.rmtree(demo_dir)
            yellowprint("Done")
            
    if started_voice:
        terminate_process_and_children(voice_handle)
        voice_handle.wait()
        yellowprint("stopped voice")
        
    stop_camera_saving()
    cpipe.done()
    topic_writer.done_session()

if __name__ == '__main__':
    global downsample

    parser = argparse.ArgumentParser()
    parser.add_argument("demo_type", help="Type of demonstration.", type=str)
    
    parser.add_argument("calib_file", help="Calibration file.", default='')
    parser.add_argument("num_cameras", help="Number of cameras in setup.", default=2, type=int)
    parser.add_argument("--downsample", help="Downsample rgbd data by factor.", default=1, type=int)
    parser.add_argument("--use_voice", help="Use voice for recording.", default=1, type=int)
    
    parser.add_argument("--single_demo", help="Single or multiple demos?", action="store_true", default=False)
    parser.add_argument("--num_demos", help="Number of demos to be recorded.", default=-1, type=int)
    parser.add_argument("--demo_name", help="Name of demo if single demo.", default="", type=str)
    args = parser.parse_args()

    downsample = args.downsample
    
    use_voice = True if args.use_voice else False
    
    if args.single_demo:
        record_single_demo (demo_type = args.demo_type, demo_name=args.demo_name,
                            calib_file = args.calib_file, num_cameras = args.num_cameras, 
                            use_voice = use_voice)
    else:
        record_pipeline (demo_type = args.demo_type, calib_file = args.calib_file, 
                         num_cameras = args.num_cameras, num_demos = args.num_demos, 
                         use_voice = use_voice)