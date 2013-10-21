import rosbag   
import cv2
import matplotlib.pyplot as ppl

from hd_utils import ros_utils as ru
from hd_utils.colorize import *

WIN_NAME = 'view_clouds'

def view_pointclouds(bag, buffer_size=3, cloud_topics=None):
    
    if cloud_topics is None:
        cloud_topics = ['/camera1/depth_registered/points',
                        '/camera2/depth_registered/points']
    
    prev_buffers = {1:[],2:[]}
    times = []
    ros_times = []
    
    for (topic, cloud, time) in bag.read_messages(topics=cloud_topics):
        
#         assert cloud.header.stamp.to_nsec() == time.to_nsec()
        
        xyz, rgb = ru.pc2xyzrgb(cloud)
        rgb = rgb.copy()
        t = cloud.header.stamp.to_sec()
        times.append(t)
        ros_times.append(time.to_sec())
        
        if topic.find('1') >= 0:
            prev_buffer = prev_buffers[1]
        else:
            prev_buffer = prev_buffers[2]
        
        if len(prev_buffer) >= buffer_size:
            prev_buffer.pop(0)
        prev_buffer.append((rgb,cloud.header.stamp.to_sec()))
        
        curr_pos = len(prev_buffer) - 1
        
        while True:
            cv2.imshow(WIN_NAME, rgb)
            print 'Time:', t
            key = cv2.waitKey()
            
            if key == 65363:
                curr_pos += 1
            elif key == 65361:
                curr_pos -= 1
            else:
                continue
            
            if curr_pos >= len(prev_buffer):
                break
            
            if curr_pos < 0:
                redprint("Reached end of buffer. Buffersize = %i"%buffer_size)
                curr_pos = 0
            
            rgb,t  = prev_buffer[curr_pos]

    ppl.plot(times,'s', markersize=5)
    ppl.hold(True)
    ppl.plot(ros_times,'ro', markersize=5)
    ppl.show()
    

def plot_times(bag, cloud_topics=None, markersize=2):
    
    if cloud_topics is None:
        cloud_topics = ['/camera1/depth_registered/points',
                        '/camera2/depth_registered/points']
    
    prev_buffers = {1:[],2:[]}
    times = []
    ros_times = []
    topic_times = {1:[],2:[]}
    topic_rostimes = {1:[],2:[]}
    
    
    
    for (topic, cloud, time) in bag.read_messages(topics=cloud_topics):
        
#         assert cloud.header.stamp.to_nsec() == time.to_nsec()
        
        ct = cloud.header.stamp.to_sec()
        rt = time.to_sec()
        times.append(ct)
        ros_times.append(rt)
        
        if topic.find('1') >= 0:
            topic_times[1].append(ct)
            topic_rostimes[1].append(rt)
        else:
            topic_times[2].append(ct)
            topic_rostimes[2].append(rt)

    ppl.plot(topic_times[1],'s', markersize=markersize)
    ppl.hold(True)
    ppl.plot(topic_times[2],'ro', markersize=markersize)
    ppl.show()
