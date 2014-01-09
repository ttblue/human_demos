# A simple class to return a time-stamped list of transforms in a serialized manner.
from __future__ import division
import numpy as np

from hd_utils.colorize import *
from hd_utils import ros_utils as ru, clouds, conversions, extraction_utils as eu
from hd_utils.defaults import asus_xtion_pro_f

import cv2

class streamize_pc():
    """
    A class that takes in a bag file and a topic. 
    Returns the latest point cloud available with that frequency. 
 
    Returns a iterator for point clouds indexed by a time-frequency:
    =============================================================
       On each call of 'next', it increments its time counter by
       1./freq and returns the last of all pointclouds b/w  t, t+1/f.
    
       If no object is present in [t,t+1/f], it returns None.

    It assumes, that time-stamps are sorted.
    Time-stamps are assumed to be numpy arrays of float.

    This class is iterable.
    """
    
    def __init__(self, bag, cloud_topics, freq, tstart=None, delay=0, speed=1.0, verbose=False):
                
        if cloud_topics == None:
            self.done = True
            return
        
        self.bag = bag
        self.verbose = verbose
        self.tstart = tstart
        self.delay = delay
        
        if not isinstance(cloud_topics, list):
            self.topics = [cloud_topics]
        else:
            self.topics = cloud_topics
        
        self.done = False
        
        self.cloud_gen = self.bag.read_messages(topics=self.topics)
        
        try:
            _,self.curr_msg,_ = self.cloud_gen.next()
            self.base_ts = self.curr_msg.header.stamp.to_sec()
        except StopIteration:
            self.done = True
            raise StopIteration ('Empty topics.')
        
        self.speed = speed
        self.dt = self.speed/freq
        self.t  = -self.dt-self.delay if self.tstart==None else self.tstart-self.base_ts-self.dt-self.delay
        self.ts = 0.0
        
        self.num_seen = 1

    def __iter__(self):
        return self
    
    def latest_time(self):
        return self.base_ts + self.ts
    
    def time_now(self):
        return self.base_ts + self.t
    
    def reset (self):
        self.cloud_gen = self.bag.read_messages(topics=self.topics)
        
        try:
            _,self.curr_msg,_ = self.cloud_gen.next()
        except StopIteration:
            self.done = True
            raise StopIteration ('Empty topics.')
                
        self.t  = -self.dt-self.delay if self.tstart==None else self.tstart-self.base_ts-self.dt-self.delay
        self.ts = 0.0
        
        self.num_seen = 1
    

    def next(self):
        if self.done:
            self.reset()
            raise StopIteration
        else:
            ttarg    = self.t + self.dt
            self.t += self.dt

            if self.ts > ttarg:
                if self.verbose:
                    print "Returning None."                
                return None
            
            msg = None
            curr_t = None
            while True:
                try:
                    _, msg, _ = self.cloud_gen.next()
                    self.num_seen += 1
                except StopIteration:
                    self.done = True
                    break
                curr_t = msg.header.stamp.to_sec() - self.base_ts
                if curr_t <= ttarg:
                    self.curr_msg = msg
                    self.ts = curr_t
                else:
                    break

            rtn_msg = self.curr_msg
            if self.verbose:
                print "Time stamp: ", self.base_ts + self.ts

            self.curr_msg = msg
            self.ts = curr_t            
            
            return rtn_msg
        
    def get_speed(self):
        return self.speed


class streamize_rgbd_pc():
    """
    A class that takes in rgbd directory . 
    Returns the latest point cloud available with that frequency. 
 
    Returns a iterator for point clouds indexed by a time-frequency:
    =============================================================
       On each call of 'next', it increments its time counter by
       1./freq and returns the last of all pointclouds b/w  t, t+1/f.
    
       If no object is present in [t,t+1/f], it returns None.

    It assumes, that time-stamps are sorted.
    Time-stamps are assumed to be numpy arrays of float.

    This class is iterable.
    """
    
    def __init__(self, rgbd_dir, frame_id, freq, tstart=None, tend=None, delay=0, speed=1.0, verbose=False):
        if rgbd_dir is None:
            self.done = True
            return

        self.rgbd_dir = rgbd_dir
        self.verbose  = verbose
        self.frame_id = frame_id
        self.tstart   = tstart
        self.tend     = tend
        self.delay    = delay

        self.rgbs_fnames, self.depth_fnames, self.stamps = eu.get_rgbd_names_times(rgbd_dir)
        self.index = 0

        self.base_ts = self.get_stamp()

        self.done = False

        self.speed = speed
        self.dt = self.speed/freq
        self.t  = -self.dt-self.delay if self.tstart==None else self.tstart-self.base_ts-self.dt-self.delay       
        self.ts = 0.0
        self.num_seen = 1


    def __iter__(self):
        return self


    def reset (self):            
        self.t  = -self.dt-self.delay if self.tstart==None else self.tstart-self.base_ts-self.dt-self.delay
        self.ts = 0.0
        self.index = 0
        self.num_seen = 1
    
    
    def get_stamp(self, index=None):
        if not index:
            index = self.index
        return self.stamps[index]

    
    def get_pc(self, index=None):
        if not index:
            index = self.index
        rgb = cv2.imread(self.rgbs_fnames[index])
        depth = cv2.imread(self.depth_fnames[index], 2)
        xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
        pc = ru.xyzrgb2pc(xyz, rgb, frame_id=self.frame_id, use_time_now=False)
        
        return pc
    
    def latest_time(self):
        return self.base_ts + self.ts
    
    def time_now(self):
        return self.base_ts + self.t
    
    def next(self):
        if self.done:
            self.reset()
            raise StopIteration
        else:
            ttarg = self.t + self.dt
            self.t += self.dt

            if self.ts > ttarg:
                if self.verbose:
                    print "Returning None."                
                return None
            
            msg = None
            msg_t = 0
            curr_t = None
            
            while True:
                if self.index == len(self.stamps) or self.ts + self.base_ts > self.tend:
                    self.done = True
                    break
                else:
                    self.index += 1
                    msg_t = self.get_stamp()
                    self.num_seen += 1
                
                curr_t = msg_t - self.base_ts;                
                
                if curr_t <= ttarg:
                    self.ts = curr_t
                else:
                    break

            rtn_msg = self.get_pc(self.index-1)
            if self.verbose:
                print "Time stamp: ", self.base_ts + self.ts

            self.ts = curr_t
            
            return rtn_msg
    
    def get_speed(self):
        return self.speed


if __name__ == '__main__':
    import rosbag, rospy, os
    from sensor_msgs.msg import PointCloud2
    
    bag = rosbag.Bag('/home/sibi/sandbox/human_demos/hd_data/demos/overhand/demo00001/demo.bag')
    
    rospy.init_node('test_pc')
    pub = rospy.Publisher('/test_pointclouds', PointCloud2)

    freq = 1
    pc_streamer = streamize_pc(bag, '/camera1/depth_registered/points', 1, tstart=1383366737.35)
    # THE ABOVE LINE WILL NOT WORK. NO MORE PC IN POINT CLOUDS.

    
    while True:
        try:
            raw_input("Hit next when ready.")
            pc = pc_streamer.next()
            if pc is None:
                redprint('No Pointcloud.')
                continue
            print pc.header.stamp.to_sec()
            pc.header.stamp = rospy.Time.now()
            print pc.header.frame_id
            pub.publish(pc)
        except StopIteration:
            break
    
    