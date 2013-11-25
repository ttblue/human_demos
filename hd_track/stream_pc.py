# A simple class to return a time-stamped list of transforms in a serialized manner.
from __future__ import division
import numpy as np

from hd_utils.colorize import *

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
    
    def __init__(self, bag, cloud_topics, freq, tstart=None, verbose=False):
        self.bag = bag
        self.verbose = verbose
        
        if not isinstance(cloud_topics, list):
            self.topics = [cloud_topics]
        else:
            self.topics = cloud_topics
        
        self.done = False
        
        self.cloud_gen = bag.read_messages(topics=self.topics)
        
        try:
            _,self.curr_msg,_ = self.cloud_gen.next()
            self.base_ts = self.curr_msg.header.stamp.to_sec()
        except StopIteration:
            self.done = True
            raise StopIteration ('Empty topics.')
                
        self.dt = 1./freq
        self.t  = -self.dt if tstart==None else tstart-self.base_ts-self.dt
        self.ts = 0.0
        
        self.num_seen = 1

    def __iter__(self):
        return self
    
    def latest_time(self):
        return self.base_ts + self.ts
    
    def time_now(self):
        return self.base_ts + self.t
    
    def next(self):
        if self.done:
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

            
if __name__ == '__main__':
    import rosbag, rospy, os
    from sensor_msgs.msg import PointCloud2
    
    bag = rosbag.Bag('/home/sibi/sandbox/human_demos/hd_data/demos/recorded/demo6.bag')
    
    rospy.init_node('test_pc')
    pub = rospy.Publisher('/test_pointclouds', PointCloud2)

    freq = 1
    pc_streamer = streamize_pc(bag, '/camera1/depth_registered/points', 1, tstart=1383366737.35)
    
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
    
    