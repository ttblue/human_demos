import time
from mayavi import mlab
import numpy as np

import mayavi_plotter as mp
from hd_utils import conversions as conv
from hd_calib import phasespace as ph

def playback_log (log_file):
    import yaml, os.path as osp
    
    log_loc = "/home/sibi/sandbox/human_demos/hd_data/phasespace_logs" 
    with open(osp.join(log_loc,log_file),"r") as fh: marker_pos = yaml.load(fh)
    
    #handle = mlab.points3d([0],[0],[0], color = (1,0,0), scale_factor = 0.25)
    #ms = handle.mlab_source
    plotter = mp.PlotterInit()
    
    prev_time = time.time()
    for step in marker_pos:
        
        markers = np.asarray(step['marker_positions'].values())
        
        clear_req = mp.gen_mlab_request(mlab.clf)
        plot_req = mp.gen_mlab_request(mlab.points3d, markers[:,0], markers[:,1], markers[:,2], color=(1,0,0),scale_factor=0.25)
            
        time_to_wait = max(step['time_stamp'] - time.time() + prev_time,0.0)
        time.sleep(time_to_wait)
        
        plotter.request(clear_req)
        plotter.request(plot_req)
        
        prev_time = time.time()
        
    print "Playback of log file %s finished."%log_file



def publish_phasespace_markers_ros (print_shit=False):

    import rospy
    import roslib; roslib.load_manifest("tf")
    import tf
    from visualization_msgs.msg import Marker, MarkerArray
    from geometry_msgs.msg import Point
    from std_msgs.msg import ColorRGBA

    ph.turn_phasespace_on()

    if rospy.get_name() == '/unnamed':
        rospy.init_node("phasespace")
    
#    ph.turn_phasespace_on()   
    marker_pub = rospy.Publisher('phasespace_markers', MarkerArray)
    tf_pub = tf.TransformBroadcaster()

    #prev_time = time.time() - 1/log_freq
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
                m.header.frame_id = ph.PHASESPACE_FRAME
                m.scale.x = m.scale.y = m.scale.z = 0.01
                m.type = Marker.CUBE
                m.color.r = 1
                m.color.a = 1
                mk.markers.append(m)
            
            trans, rot = None, None
            try:
                trans, rot = conv.hmat_to_trans_rot(ph.marker_transform(0,1,2, marker_pos))
            except:
                print "Could not find phasespace transform."
                pass
            
            #curr_time = time.time()
            #if curr_time - prev_time > 1/log_freq:
            if print_shit: print marker_pos
            marker_pub.publish(mk)
            if trans is not None:
                tf_pub.sendTransform(trans, rot, rospy.Time.now(), "ps_marker_transform", ph.PHASESPACE_FRAME)
            #    prev_time = curr_time
                
            
            #sleep for remainder of the time
            #time_passed = rospy.Time.now().to_sec() - time_stamp.to_sec()
            #time.sleep(0.2)
            #time.sleep(max(1/log_freq-time_passed,0))
        except KeyboardInterrupt:
            break

    ph.turn_phasespace_off()
