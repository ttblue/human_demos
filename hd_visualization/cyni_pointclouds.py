#!/usr/bin/ipython -i
import rospy, time
from sensor_msgs.msg import PointCloud2

import cv2
import cyni

from hd_utils import clouds, ros_utils as ru
from hd_utils.yes_or_no import yes_or_no

asus_xtion_pro_f = 544.260779961

# Call cyni.initialize() before this.
def get_device (device_id):
    """
    Takes in a string @device_id. If device_id is 
    """
    
    devices = cyni.enumerateDevices()
    if len(devices) == 0:
        print "No devices found! Maybe you forgot to call cyni.initialize()"
        return None
    
    if device_id[0] == "#":
        try:
            num = int(device_id[1:])
            if num > len(devices):
                print "Index out of range."
                return None
            else: return cyni.Device(devices[num-1]['uri'])
        except:
            print "Incorrect device_id %s"%device_id
            return None
    else:
        device_uris = [device['uri'] for device in devices]
        if device_id not in device_uris:
            print "Device with URI %s not found."%device_id
            return None
        else: return cyni.Device(device_id)

CYNI_INITIALIZED = False

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

def get_streams(device_id="#1"):
    global CYNI_INITIALIZED
    if not CYNI_INITIALIZED:
        cyni.initialize()
        CYNI_INITIALIZED = True
        
    device = get_device(device_id)
    device.open()
    depthStream = device.createStream("depth", width=640, height=480, fps=30)
    colorStream = device.createStream("color", width=640, height=480, fps=30)
    device.setImageRegistrationMode("depth_to_color")
    device.setDepthColorSyncEnabled(on=True)
    
    return {"device": device, "depth": depthStream, "color": colorStream}

NUM_CAMERAS = 1
TOGGLE_FREQ = 1.0

def visualize_pointcloud():
    """
    Visualize point clouds from cyni data.
    """
    if rospy.get_name() == '/unnamed':
        rospy.init_node("visualize_pointcloud")
    
    camera_frame="camera_depth_optical_frame"
    
    pc_pubs = []
    sleeper = rospy.Rate(30)
    
    streams = []
    for i in xrange(1, NUM_CAMERAS+1):
        cam_streams = get_streams("#%d"%i)
        if cam_streams is not None:
            streams.append(cam_streams)
            cam_streams["depth"].start()
            cam_streams["depth"].setEmitterState(False)
            cam_streams["color"].start()
            pc_pubs.append(rospy.Publisher("camera_depth_registered_points"%i, PointCloud2))
        else: break
    
    indiv_freq = TOGGLE_FREQ/NUM_CAMERAS
    
    print "Streaming now: Pointclouds only."
    try:
        while True:
            print "Publishing data..."
            for (i,stream) in enumerate(streams):
                stream["depth"].setEmitterState(True)
                depth = stream["depth"].readFrame().data
                rgb = cv2.cvtColor(stream["color"].readFrame().data, cv2.COLOR_RGB2BGR)
                
                
#                pc = ru.xyzrgb2pc(clouds.depth_to_xyz(depth, asus_xtion_pro_f), rgb, camera_frame)
#                pc_pubs[i].publish(pc)
                time.sleep(1/indiv_freq*0.5)
                stream["depth"].setEmitterState(False)
                time.sleep(1/indiv_freq*0.5)
                
            if yes_or_no("Done?"): break
            #wait for a bit?
    except KeyboardInterrupt:
        print "Keyboard interrupt. Exiting."