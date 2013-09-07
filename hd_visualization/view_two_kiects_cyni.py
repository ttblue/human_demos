#!/usr/bin/env python
import  cv2
import numpy as np
import subprocess
import cyni
import Image
import time

#subprocess.call("sudo killall XnSensorServer", shell=True)

cmap = np.zeros((256, 3),dtype='uint8')
cmap[:,0] = range(256)
cmap[:,2] = range(256)[::-1]
cmap[0] = [0,0,0]

#grabber = cloudprocpy.CloudGrabber("#1")
#grabber.startRGBD()
#g1 = cloudprocpy.CloudGrabber("#1")
#g1.startRGBD()


cyni.initialize()
device_list = cyni.enumerateDevices()
d1 = cyni.Device(device_list[0]['uri'])
d2 = cyni.Device(device_list[1]['uri'])
d1.open()
d2.open()
#colorStream = d1.createStream("color", width=640, height=480, fps=30)
#colorStream.start()
ds1 = d1.createStream("depth", width=640, height = 480, fps=30)
ds1.start()
ds2 = d2.createStream("depth", width=640, height = 480, fps = 30)
ds2.start()
emitterControl = False
try:
    while True:
        
        #rgb = colorStream.readFrame()
        #cv2.imshow("rgb", rgb.data)
        #start = time.time()
        #emitterControl = not emitterControl
        #print emitterControl
        #ds1.setEmitterState(emitterControl)
        #ds2.setEmitterState(not emitterControl)
        depth1 = ds1.readFrame()
        depth2 = ds2.readFrame()
        #print depth.data
        #while np.sum(depth.data) != 0:
        #    depth = depthStream.readFrame()
        #    print depth.data
        #print '%f seconds'%(time.time() - start)
        #time.sleep(1)
        cv2.imshow("d1", cmap[np.fmin((depth1.data*.064).astype('int'), 255)])
        cv2.imshow("d2", cmap[np.fmin((depth2.data*.064).astype('int'), 255)])
        #cv2.imshow("depth", depth.data)
        time.sleep(1)
        #depthStream.setEmitterState(False)
        cv2.waitKey(30)
        #depthStream.setEmitterState(False)
        #emitterControl = not emitterControl
        

        #r1, d1 = g1.getRGBD()
        #cv2.imshow("rgb", rgb)
        #cv2.imshow("r1", r1)
        #cv.SetMouseCallback("rgb", on_mouse, 0)
        #cv2.imshow("depth", cmap[np.fmin((depth*.064).astype('int'), 255)])
        #cv2.waitKey(30)
        #break
except KeyboardInterrupt:
    print "got Control-C"
