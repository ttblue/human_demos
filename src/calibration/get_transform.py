#!/usr/bin/env python


from rapprentice import PR2, berkeley_pr2, clouds
import cloudprocpy, cv, cv2, numpy as np
import subprocess 
import sys
from OWL import *


MARKER_COUNT = 4
SERVER_NAME = "192.168.1.126"
INIT_FLAGS = 0

def get_sensor_coordinates_kinect():
    subprocess.call("sudo killall XnSensorServer", shell=True)
    grabber = cloudprocpy.CloudGrabber()
    grabber.startRGBD()
    rgb, depth = grabber.getRGBD()
    clicks = []

    def mouse_click(event, x, y, flags, params):
        if event == cv.CV_EVENT_LBUTTONDOWN:
            clicks.append([x, y])

    # rewrite this
    def find_nearest(xy, indicators):
        nearest = []
        for (x, y) in xy:
            min_dist = 10000000
            closest_point = [0, 0]
            for j in xrange(len(indicators)):
                for i in xrange(len(indicators[0])):
                    dist = (x - i) ** 2 + (y - j) ** 2
                    if indicators[j][i] and dist < min_dist:
                        #print 'satisfied'
                        min_dist = dist
                        closest_point = [i, j]
            nearest.append(closest_point)
        return nearest
    
    def find_avg(points, clicks, cloud):
        avgPoints = []
        rad = 0.02
        r = 5
        for (x,y), point in zip(clicks,points):
            avg = np.array([0,0,0])
            num = 0
            for i in range(-r,r+1):
                for j in range(-r,r+1):
                    cloudPt = cloud[y+i, x+j];
                    if not np.isnan(cloudPt).any() and np.linalg.norm(point - cloudPt) < rad:
                        avg += cloudPt
                        num += 1
            if num == 0:
                avgPoints.append(point)
            else:
                avgPoints.append(avg / num)
        return avgPoints       
    


    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    red_mask = ((h<10) | (h>150)) & (s > 100) & (v > 100)
    valid = depth*(depth > 0)
    xyz_k = clouds.depth_to_xyz(valid, berkeley_pr2.f)
    
    clickPoints = []
    #get click points
    for (x, y) in clicks:
        clickPoints.append(xyz_k[y][x])
        
    #cv2.imshow("red",red_mask.astype('uint8')*255)
    cv2.imshow("red", red_mask.astype('uint8')*255)
    cv2.imshow("rgb", rgb)
    cv.SetMouseCallback("rgb", mouse_click, 0)
    print "press enter to continue"
    cv2.waitKey()
    
    red_mask3d = np.tile(np.atleast_3d(red_mask), [1,1,3])
    np.putmask (xyz_k, red_mask3d != 1, np.NAN) 

    

    avgPoints = find_avg(clicks, clickPoints, xyz_k)
    print clicks
    print clickPoints
    print avgPoints
    return avgPoints
    

def turn_phasespace_on():
    if(owlInit(SERVER_NAME, INIT_FLAGS) < 0):
        print "init error: ", owlGetError()
        sys.exit(0)

    # create tracker 0
    tracker = 0
    owlTrackeri(tracker, OWL_CREATE, OWL_POINT_TRACKER)

    # set markers
    for i in range(MARKER_COUNT):
        owlMarkeri(MARKER(tracker, i), OWL_SET_LED, i)

    # activate tracker
    owlTracker(tracker, OWL_ENABLE)
    
    #return
    if(owlGetStatus() == 0):
        owl_print_error("error in point tracker setup", owlGetError())
        sys.exit(0)
    owlSetFloat(OWL_FREQUENCY, OWL_MAX_FREQUENCY)
    owlSetInteger(OWL_STREAMING, OWL_ENABLE)

def turn_phasespace_off():
    owlDone()

def get_sensor_coordinates_phasespace():
    sensor_locations = []
    while(1):
        markers = []
        n = owlGetMarkers(markers, 50)
        err = owlGetError()
        if (err != OWL_NO_ERROR):
            break
        if(n==0): continue
        if(n == 4):
            for i in range(n):
                #print "%d) %.2f %.2f %.2f" % (i, markers[i].x, markers[i].y, markers[i].z)
                sensor_locations.append([markers[i].x, markers[i].y, markers[i].z])
            break


    return sensor_locations

def main():    
    turn_phasespace_on()
    kinect = get_sensor_coordinates_kinect()
    phasespace = get_sensor_coordinates_phasespace()
    turn_phasespace_off()
    print kinect
    print phasespace



main()



