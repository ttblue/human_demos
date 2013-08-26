#/usr/bin/ipython -i
import cv2
import cyni

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


def initialize_cyni(device_id="#1"):
    
    cyni.initialize()
    device = cyni.getAnyDevice()#get_device(device_id)
    device.open()
    depthStream = device.createStream("depth", width=640, height=480, fps=30)
    colorStream = device.createStream("color", width=640, height=480, fps=30)
    device.setImageRegistrationMode("depth_to_color")
    device.setDepthColorSyncEnabled(on=True)
    
    return depthStream, colorStream


def visualize_pointcloud():
    """
    Visualize point clouds from cyni data.
    """
    depthStream.start()
    colorStream.start()
    
    print "Streaming now: Pointclouds only."
    while True:
        try:
            
            
            pc = ru.xyzrgb2pc(clouds.depth_to_xyz(d, asus_xtion_pro_f), r, camera_frame)
            pc_pub.publish(pc)

            sleeper.sleep()
        except KeyboardInterrupt:
            print "Keyboard interrupt. Exiting."
            break