#!/usr/bin/env python
from rapprentice import berkeley_pr2, clouds
import phasespace as ph

import cloudprocpy, cv, cv2, numpy as np
import subprocess, sys

import OWL as owl


def get_markers_kinect():
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
            num = 0.0
            for i in range(-r,r+1):
                for j in range(-r,r+1):
                    cloudPt = cloud[y+i, x+j,:];
                    if not np.isnan(cloudPt).any() and np.linalg.norm(point - cloudPt) < rad:
                        avg = avg + cloudPt
                        num += 1.0
            if num == 0.0:
                print "not found"
                avgPoints.append(point)
            else:
                print "found"
                avgPoints.append(avg / num)
        return avgPoints       
    


    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    red_mask = ((h<10) | (h>150)) & (s > 100) & (v > 100)
    #valid = depth*(depth > 0)
    xyz_k = clouds.depth_to_xyz(depth, berkeley_pr2.f)
            
    #cv2.imshow("red",red_mask.astype('uint8')*255)
    cv2.imshow("red", red_mask.astype('uint8')*255)
    cv2.imshow("rgb", rgb)
    cv.SetMouseCallback("rgb", mouse_click, 0)
    print "press enter to continue"
    cv2.waitKey()
    
    clickPoints = []
    #get click points
    for (x, y) in clicks:
        clickPoints.append(np.copy(xyz_k[y][x]))
    
#     import IPython
#     IPython.embed()
    red_mask3d = np.tile(np.atleast_3d(red_mask), [1,1,3])
    np.putmask (xyz_k, red_mask3d != 1, np.NAN)     

    print clicks
    print clickPoints
    avgPoints = find_avg(clickPoints, clicks, xyz_k)
    print avgPoints
    return avgPoints
    
def find_rigid_tfm (kin_points, ps_points, homogeneous=True):
    kin_points = np.asarray(kin_points)
    kin_points = kin_points[ps_points.keys()]
    ps_points = np.asarray(ps_points.values())
    
    if kin_points.shape != ps_points.shape:
        print "Not the same number of points"
        return
    elif len(kin_points) < 3:
        print "Not enough points"
        return
    
    kin_center = kin_points.sum(axis=0)/float(kin_points.shape[0])
    ps_center = ps_points.sum(axis=0)/float(ps_points.shape[0])
    
    X = kin_points - kin_center
    Y = ps_points - ps_center
    
    S = X.T.dot(Y)
    # svd gives U, Sigma and V.T
    U, Sig, V = np.linalg.svd(S, full_matrices=True)

    ref_rot = np.eye(3,3)
    ref_rot[2,2] = np.round(np.linalg.det(V.dot(U.T)))
    
    import IPython
    IPython.embed()   
    
    R = V.T.dot(ref_rot.dot(U.T))
    t = ps_center - R.dot(kin_center)
    
    if homogeneous:
        Tfm = np.eye(4,4)
        Tfm[0:3,0:3] = R
        Tfm[0:3,3] = t
        return Tfm
    else:
        return R,t
    
def get_sensor_coordinates_phasespace():
    sensor_locations = []
    while(1):
        markers = []
        n = owl.owlGetMarkers(markers, 50)
        err = owl.owlGetError()
        if (err != owl.OWL_NO_ERROR):
            break
        if(n==0): continue
        if(n == 4):
            for i in range(n):
                #print "%d) %.2f %.2f %.2f" % (i, markers[i].x, markers[i].y, markers[i].z)
                sensor_locations.append([markers[i].x, markers[i].y, markers[i].z])
            break


    return sensor_locations


def main():    
    ph.turn_phasespace_on()
    #kinect = get_sensor_coordinates_kinect()
    #phs = ph.get_marker_positions()
    phs = get_sensor_coordinates_phasespace()
    ph.turn_phasespace_off()
    #print kinect
    print phs


def test_rigidtfm():
    X = np.array([[1,0,0],[0,2,0],[0,0,3],[1,1,1]])
    Y = np.copy(X)
    Y_d = {}
    i = 0
    for y in Y:
        Y_d[i] = y
        i += 1
    
    print find_rigid_tfm(X,Y_d)
#main()