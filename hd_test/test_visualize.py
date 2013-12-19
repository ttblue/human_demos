#!/usr/bin/ipython -i
import cv2, subprocess
import roslib; roslib.load_manifest('tf')
import rospy, tf
from sensor_msgs.msg import PointCloud2

roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse
import cloudprocpy as cpr
from hd_utils import conversions, ros_utils as ru, clouds
from hd_utils.defaults import asus_xtion_pro_f

getMarkers = None
req = MarkerPositionsRequest()

WIN_NAME="click_win"

def get_ar_transform_id (xyz, rgb, idm=None):    
    """
    In order to run this, ar_marker_service needs to be running.
    """
    req.pc = ru.xyzrgb2pc(xyz, rgb, '/camera_link')    
    res = getMarkers(req)
    
    marker_tfm = {marker.id:conversions.pose_to_hmat(marker.pose.pose) for marker in res.markers.markers}
    
    if not idm: return marker_tfm
    if idm not in marker_tfm: return None
    return marker_tfm[idm]

def get_ar_transform_id_unorganized (xyz, rgb, idm=None):    
    """
    In order to run this, ar_marker_service needs to be running.
    """
    
    print xyz
    print rgb
    
    xyz = xyz.reshape(xyz.shape[0]*xyz.shape[1], xyz.shape[2])
    rgb = rgb.reshape(rgb.shape[0]*rgb.shape[1], rgb.shape[2])
    
    print xyz
    print rgb
    
    req.pc = ru.xyzrgb2pc(xyz, rgb, '/camera_link')    
    res = getMarkers(req)
    
    marker_tfm = {marker.id:conversions.pose_to_hmat(marker.pose.pose) for marker in res.markers.markers}
    
    if not idm: return marker_tfm
    if idm not in marker_tfm: return None
    return marker_tfm[idm]

def try_smaller_pc():
    global getMarkers
    
    rospy.init_node('test_ar')
    subprocess.call("killall XnSensorServer", shell=True)
    
    grabber = cpr.CloudGrabber()
    grabber.startRGBD()
    
    getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)
    
    class clickClass ():
        x = None
        y = None
        done = False
        
        def callback(self, event, x, y, flags, param):
            if self.done:
                return
            elif event == cv2.EVENT_LBUTTONDOWN:
                self.x = x
                self.y = y
                self.done = True
   
   
    range = 100
    
    try:
        while True:
            rgb, depth = grabber.getRGBD()
            xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
            
            click = clickClass() 
            print "Click on the point at the middle of the AR marker"
            
            cv2.imshow(WIN_NAME, rgb)
            cv2.setMouseCallback(WIN_NAME, click.callback)
            while not click.done:
                cv2.waitKey(100)
                
            x = click.x
            y = click.y
            
            bl = (max(0,x-range), max(0,y-range))
            tr = (min(rgb.shape[1]-1,x+range), min(rgb.shape[0]-1, y+range))
            
            cv2.circle(rgb, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(rgb, tr, 5, (255, 0, 0), -1)
            cv2.circle(rgb, bl, 5, (0, 255, 0), -1)
            cv2.rectangle(rgb, bl, tr, (0, 0, 255), 1)
            cv2.imshow(WIN_NAME, rgb)    
            cv2.waitKey(3000)
            
            checkRGB = rgb[bl[0]:tr[0],bl[1]:tr[1],:]
            checkXYZ = xyz[bl[0]:tr[0],bl[1]:tr[1],:]
            
            tfms = get_ar_transform_id_unorganized(checkXYZ, checkRGB)
            print "The found AR markers are:"
            print tfms
    except KeyboardInterrupt:
        print "Done"