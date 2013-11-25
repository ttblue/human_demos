#!/usr/bin/env python
import cv2
import numpy as np
import subprocess
import cyni
import Image
import time
import subprocess
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True, type=str)
parser.add_argument('--freq', required=True, type=float)
args = parser.parse_args()
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
data = []
start = time.time()
i = 0
try:
    while True:
        #start = time.time()
        depthStream.setEmitterState(True)
        #print 'time start %f'%(time.time() - start)
        #print time.time() - start
        #depth = depthStream.readFrame()
        #while np.sum(depth.data) == 0:
        #    depth = depthStream.readFrame()
        time.sleep(1/args.freq)
        #start = time.time()
        depth = depthStream.readFrame()
        data.append(depth.data)
        #print 'time read %f'%(time.time() - start)
        #print 1
	#print depth.data
        #data.append(depth.data)
        #start = time.time()
        depthStream.setEmitterState(False)
        #depth = depthStream.readFrame()
        #while np.sum(depth.data) != 0:
        #    depth = depthStream.readFrame()
        #print 'time %f'%(time.time() - start)
        #print depth.data
        #time.sleep(0.5/args.freq)
        #cv2.imshow("depth", cmap[np.fmin((depth.data*.064).astype('int'), 255)])
        #cv2.waitKey(100)
except KeyboardInterrupt:
    print "saving" 
    subprocess.call("rm -rf depth/%s"%(args.name), shell=True)
    subprocess.call("mkdir depth/%s"%(args.name), shell=True)
    for i in xrange(len(data)):
        Image.fromarray(cyni.depthMapToImage(data[i])).save("depth/%s/depth%i.png"%(args.name, i))
    t = time.time() - start
    print len(data)/t
