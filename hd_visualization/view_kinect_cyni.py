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


cyni.initialize()

device = cyni.getAnyDevice()
#subprocess.call("sudo killall XnSensorServer", shell=True)
device.open()
colorStream = device.createStream("color", width=640, height=480, fps=30)
colorStream.start()
depthStream = device.createStream("depth", width=640, height = 480, fps=30)
depthStream.start()
try:
    while True:
        
        rgb = colorStream.readFrame()
        cv2.imshow("rgb", rgb.data)
        depth = depthStream.readFrame()
        cv2.imshow("depth", cmap[np.fmin((depth.data*.064).astype('int'), 255)])
        cv2.waitKey(30)
       
except KeyboardInterrupt:
    print "got Control-C"
