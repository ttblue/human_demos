import itertools
import os.path as osp, time

from  hd_calib import phasespace as ph, get_transform as gt

log_freq = 5.0

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
    
    ph.turn_phasespace_on()   
    marker_pub = rospy.Publisher('phasespace_markers', MarkerArray)

    while True:
        try:
            marker_pos = ph.get_marker_positions()
            time_stamp = rospy.Time.now()
            
            mk = MarkerArray()
            for i in marker_pos:
                m = Marker()
                m.pose.position.x = marker_pos[i].x
                m.pose.position.y = marker_pos[i].y
                m.pose.position.z = marker_pos[i].z
                m.id = marker_pos[i].id
                m.header.stamp = time_stamp
                m.header.frame_id = "phasespace_frame"
                
                mk.markers.append(m)
                
            marker_pub.publish(mk)
            
            #sleep for remainder of the time
            time_passed = rospy.Time.now().to_sec() - time_stamp.to_sec()
            time.sleep(max(1/log_freq-time_passed,0))
        except KeyboardInterrupt:
            break

    ph.turn_phasespace_off()

def initialize_ros_logging(tfm_file=None):
    
    import yaml
    
    ph.turn_phasespace_on()
    kin_points = gt.get_markers_kinect()
    ps_points = ph.get_marker_positions()
    
    Tfm = gt.find_rigid_tfm(kin_points, ps_points)
    log_loc = "/home/sibi/sandbox/human_demos/hd_data/transforms"
    if tfm_file == None:
        base_name = "tfm"
        file_base = osp.join(log_loc, base_name)
        for suffix in itertools.chain("", (str(i) for i in itertools.count())):
            if not osp.isfile(file_base+suffix+'.txt'):
                file_name = file_base+suffix+'.txt'
                with open(file_name,"w") as fh: yaml.dump(Tfm.tolist())
                break
    else:
        file_name = osp.join(file_base, tfm_file)
        with open(file_name,"w") as fh: yaml.dump(Tfm.tolist())
        
    
    publish_phasespace_markers_ros()
