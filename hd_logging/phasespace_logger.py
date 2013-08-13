#!/usr/bin/ipython -i
import itertools
import os.path as osp, time, subprocess
from multiprocessing import Process

from  hd_calib import phasespace as ph, get_transform as gt
from rapprentice import conversions as conv

log_freq = 20.0

def log_phasespace_markers (file_name=None):
    
    log_loc = "/home/sibi/sandbox/human_demos/hd_data/phasespace_logs"
    if file_name == None:
        base_name = "phasespace_log"    
        file_base = osp.join(log_loc, base_name)
        for suffix in itertools.chain("", (str(i) for i in itertools.count())):
            if not osp.isfile(file_base+suffix+'.log'):
                file_name = file_base+suffix+'.log'
                with open(file_name,"w") as fh: fh.write('')
                break
    else:
        file_name = osp.join(file_base, file_name)
        with open(file_name,"w") as fh: fh.write('')

    ph.turn_phasespace_on()
    start_time = time.time()

    while True:
        try:
            marker_pos = ph.get_marker_positions()
            time_stamp = time.time() - start_time
        except KeyboardInterrupt:
            break
        
        with open(file_name, "a") as fh:
            fh.write("- time_stamp: %f\n"%time_stamp)
            fh.write("  marker_positions: \n")
            for id in marker_pos:
                fh.write("   %i: "%id+str(marker_pos[id]) + "\n")

        try:
            #sleep for remainder of the time
            wait_time = time.time() - start_time - time_stamp
            time.sleep(max(1/log_freq-time.time()+start_time+time_stamp,0))
        except KeyboardInterrupt:
            break

    ph.turn_phasespace_off()
    print "Finished logging to file: "+file_name


def publish_phasespace_markers_ros ():

    import rospy
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA

    if rospy.get_name() == '/unnamed':
        rospy.init_node("phasespace")
    
#    ph.turn_phasespace_on()   
    marker_pub = rospy.Publisher('phasespace_markers', MarkerArray)

    prev_time = time.time() - 1/log_freq
    while True:
        try:
            marker_pos = ph.get_marker_positions() 
            #print marker_pos#.keys()
            time_stamp = rospy.Time.now()

            mk = MarkerArray()
            for i in marker_pos:
                m = Marker()
                m.pose.position.x = marker_pos[i][0]
                m.pose.position.y = marker_pos[i][1]
                m.pose.position.z = marker_pos[i][2]
                m.pose.orientation.w = 1
                m.id = i
                m.header.stamp = time_stamp
                m.header.frame_id = "phasespace_frame"
                m.scale.x = m.scale.y = m.scale.z = 0.01
                m.type = Marker.CUBE
                m.color.r = 1
                m.color.a = 1
                mk.markers.append(m)
                
            curr_time = time.time()
            if curr_time - prev_time > 1/log_freq:
                print marker_pos
                marker_pub.publish(mk)
                prev_time = curr_time
                
            
            #sleep for remainder of the time
            #time_passed = rospy.Time.now().to_sec() - time_stamp.to_sec()
            #time.sleep(0.2)
            #time.sleep(max(1/log_freq-time_passed,0))
        except KeyboardInterrupt:
            break

    ph.turn_phasespace_off()



from threading import Thread
import roslib; roslib.load_manifest("tf")
import rospy, tf
import numpy as np

class transformPublisher (Thread):
    
    def __init__(self, Tfm):
        Thread.__init__(self)
        self.trans, self.rot = conv.hmat_to_trans_rot(np.linalg.inv(Tfm))
        self.transform_broadcaster = tf.TransformBroadcaster()
        self.parent_frame = "/camera_depth_optical_frame"
        self.child_frame = "/phasespace_frame"
        self.time_period = 0.001

    def run (self):
        while True:
            self.transform_broadcaster.sendTransform(self.trans, self.rot,
                                                     rospy.Time.now(),
                                                     self.child_frame,
                                                     self.parent_frame)
            time.sleep(self.time_period)


#def static_tfm_publisher (Tfm):
#    trans, rot = conv.hmat_to_trans_rot(Tfm)
    #subprocess.call("")
#    subprocess.call("rosrun tf static_transform_publisher %f %f %f %f %f %f %f /phasespace_frame /camera_depth_frame 100"
#                        %(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]), shell=True)

def initialize_ros_logging(tfm_file=None):
    
#    import yaml
    
    rospy.init_node("ros_logger")
    
    ph.turn_phasespace_off()
    ph.turn_phasespace_on()
    print "Getting kinect points"
    kin_points = gt.get_markers_kinect()
    print "Kinect points: ",kin_points

    print "Getting marker points"
    ps_points = ph.get_marker_positions()
    print "Phasespace points: ",ps_points
    
    Tfm = gt.find_rigid_tfm(kin_points, ps_points)
    print "Transform:", Tfm
#    ph.turn_phasespace_off()    
    
    
#    process = Process(target=static_tfm_publisher, args=(Tfm,))
#    process.start()
    tfm_pub = transformPublisher(Tfm)
    tfm_pub.start()
    
#     log_loc = "/home/sibi/sandbox/human_demos/hd_data/transforms"
#     if tfm_file == None:
#         base_name = "tfm"
#         file_base = osp.join(log_loc, base_name)
#         for suffix in itertools.chain("", (str(i) for i in itertools.count())):
#             if not osp.isfile(file_base+suffix+'.txt'):
#                 file_name = file_base+suffix+'.txt'
#                 with open(file_name,"w") as fh: yaml.dump(Tfm.tolist())
#                 break
#     else:
#         file_name = osp.join(file_base, tfm_file)
#         with open(file_name,"w") as fh: yaml.dump(Tfm.tolist())
#             
    subprocess.call("killall XnSensorServer", shell=True)
    publish_phasespace_markers_ros()
