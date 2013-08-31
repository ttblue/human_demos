#!/usr/bin/env python
import cv2
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

print 0
cyni.initialize()
print 1
device = cyni.getAnyDevice()
print 2
device.open()
print 3
#colorStream = device.createStream("color", width=640, height=480, fps=30)
#colorStream.start()
depthStream = device.createStream("depth", width=640, height = 480, fps=30)
depthStream.start()
emitterControl = False
try:
    while True:
        
        #rgb = colorStream.readFrame()
        #cv2.imshow("rgb", rgb.data)
        #start = time.time()
        emitterControl = not emitterControl
        print emitterControl
        depthStream.setEmitterState(emitterControl)
        depth = depthStream.readFrame()
        print depth.data
        #while np.sum(depth.data) != 0:
        #    depth = depthStream.readFrame()
        #    print depth.data
        #print '%f seconds'%(time.time() - start)
        #time.sleep(1)
        cv2.imshow("depth", cmap[np.fmin((depth.data*.064).astype('int'), 255)])
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
