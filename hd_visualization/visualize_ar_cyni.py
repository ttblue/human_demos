#!/usr/bin/ipython -i
#import cloudprocpy as cpr
import cyni
import roslib
roslib.load_manifest('tf')
import rospy, tf
roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse
from sensor_msgs.msg import PointCloud2

from hd_utils import conversions, ros_utils as ru, clouds
from hd_utils.defaults import asus_xtion_pro_f


getMarkers = None
req = MarkerPositionsRequest()

def get_ar_transform_id (depth, rgb, idm=None):    
    """
    In order to run this, ar_marker_service needs to be running.
    """
    req.pc = ru.xyzrgb2pc(clouds.depth_to_xyz(depth, asus_xtion_pro_f), rgb, '/camera_link')    
    res = getMarkers(req)
    
    marker_tfm = {marker.id:conversions.pose_to_hmat(marker.pose.pose) for marker in res.markers.markers}
    
    if not idm: return marker_tfm
    if idm not in marker_tfm: return None
    return marker_tfm[idm]

if __name__ == '__main__':
    """
    Visualize point clouds from openni grabber and AR marker transforms through ROS.
    """
    global getMarkers
    
    if rospy.get_name() == '/unnamed':
        rospy.init_node("visualize_ar_cyni")
    
    camera_frame = "camera_depth_optical_frame"
    
    tf_pub = tf.TransformBroadcaster()
    pc_pub = rospy.Publisher("camera_points", PointCloud2)
    sleeper = rospy.Rate(30)
    
    getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)
    
    cyni.initialize()

    device = cyni.getAnyDevice()
    #subprocess.call("sudo killall XnSensorServer", shell=True)
    device.open()
    colorStream = device.createStream("color", width=640, height=480, fps=30)
    colorStream.start()
    depthStream = device.createStream("depth", width=640, height = 480, fps=30)
    depthStream.start()
    
    print "Streaming now."
    while True:
        try:
            r = colorStream.readFrame().data
            d = depthStream.readFrame().data
            ar_tfm = get_ar_transform_id (d, r, 2)
            print ar_tfm
            #pc = ru.xyzrgb2pc(clouds.depth_to_xyz(d, asus_xtion_pro_f), r, camera_frame)
            #pc_pub.publish(pc)
                        
            
            try:
                trans, rot = conversions.hmat_to_trans_rot(ar_tfm)
                tf_pub.sendTransform(trans, rot, rospy.Time.now(), "ar_marker_%d"%i, camera_frame)
            except:
                print "Warning: Problem with AR transform."               
                pass
            
            sleeper.sleep()
        except KeyboardInterrupt:
            print "Keyboard interrupt. Exiting."
            break
        
