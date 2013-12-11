#!/usr/bin/env python
from hd_utils import clouds, ros_utils
import phasespace as ph

import cloudprocpy, cv, cv2, numpy as np
import subprocess, sys

import OWL as owl

asus_xtion_pro_f = 544.260779961

def get_markers_kinect():
    subprocess.call("killall XnSensorServer", shell=True)
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
    
    def find_nearest_radius(points, clicks, cloud):
        markerPoints = []
        rad = 0.01
        r = 5
        for (x,y), point in zip(clicks,points):
            
            checkPoints = cloud[max(0,y-r):min(cloud.shape[0]-1,y+r), max(0,x-r):min(cloud.shape[1]-1, x+r),:]
            np.putmask(checkPoints, np.isnan(checkPoints), np.infty)
            dist = checkPoints - point[None,None,:]
            dist = np.sum((dist*dist), axis=2)
            yi,xi = np.unravel_index(np.argmin(dist), dist.shape)
            
            point =  checkPoints[yi,xi]
            if np.isnan(point).any() or np.min(dist) > rad:
                print "No point found for clicked point: ", point
            else:
                markerPoints.append(point)
            
        return markerPoints

    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    red_mask = (v>150) & ((h<10) | (h>150))# & (v > 200)# & (s > 100)
    #valid = depth*(depth > 0)
    xyz_k = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
            
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

    markerPoints = find_nearest_radius(clickPoints, clicks, xyz_k)
    return markerPoints

def get_markers_kinect_ros():
    
    import rospy
    from sensor_msgs.msg import PointCloud2
    rospy.init_node('get_pc')
    
    class msg_storer:
        def __init__(self):
            self.last_msg = None
        def callback(self,msg):
            self.last_msg = msg
        
    def mouse_click(event, x, y, flags, params):
        if event == cv.CV_EVENT_LBUTTONDOWN:
            clicks.append([x, y])

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

        
    pc_storer = msg_storer()
    rospy.Subscriber('/camera/depth_registered/points', PointCloud2, callback=pc_storer.callback)
    
    wait = rospy.Rate(10)
    while pc_storer.last_msg == None:
        print "waiting for point cloud"
        wait.sleep()

    pc = pc_storer.last_msg
    xyz, rgb = ros_utils.pc2xyzrgb(pc)
    arr = np.fromstring(pc.data,dtype='float32').reshape(pc.height,pc.width,pc.point_step/4)
    rgb0 = np.ndarray(buffer=arr[:,:,4].copy(),shape=(pc.height, pc.width,4),dtype='uint8')
    print arr.shape, arr.dtype
    print rgb0.shape, rgb0.dtype
    print xyz.shape, xyz.dtype
    print rgb.shape, rgb.dtype
    clicks = []

    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]

    red_mask = ((h<10) | (h>150)) & (s > 100) & (v > 100)
            
    #cv2.imshow("red",red_mask.astype('uint8')*255)
    cv2.imshow("red", red_mask.astype('uint8')*255)
    cv2.imshow("rgb", rgb)
    cv.SetMouseCallback("rgb", mouse_click, 0)
    print "press enter to continue"
    cv2.waitKey()
    
    clickPoints = []
    #get click points
    for (x, y) in clicks:
        clickPoints.append(np.copy(xyz[y][x]))
    
#     import IPython
#     IPython.embed()
    red_mask3d = np.tile(np.atleast_3d(red_mask), [1,1,3])
    np.putmask (xyz, red_mask3d != 1, np.NAN)     

    print clicks
    print clickPoints
    avgPoints = find_avg(clickPoints, clicks, xyz)
    print avgPoints
    return avgPoints

    
def find_rigid_tfm (points1, points2, homogeneous=True):
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    
    if points1.shape != points2.shape:
        print "Not the same number of points"
        return
    elif points1.shape[0] < 3:
        print "Not enough points"
        return
    
    center1 = points1.sum(axis=0)/float(points1.shape[0])
    center2 = points2.sum(axis=0)/float(points2.shape[0])
    
    X = points1 - center1
    Y = points2 - center2
    
    S = X.T.dot(Y)
    # svd gives U, Sigma and V.T
    U, Sig, V = np.linalg.svd(S, full_matrices=True)

    ref_rot = np.eye(3,3)
    ref_rot[2,2] = np.round(np.linalg.det(V.dot(U.T)))
    
#     import IPython
#     IPython.embed()   
    
    R = V.T.dot(ref_rot.dot(U.T))
    t = center2 - R.dot(center2)
    
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

def rotation_matrix(axis,theta):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


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

    print find_rigid_tfm(X,Y)
    
def test_rigidtfm2():
    X = np.array([[1,0,0],[0,2,0],[0,0,3],[1,1,1]])
    R = rotation_matrix(np.array([1,2.2,3]),2.1)
    t = np.array([3,4.4,3])
    Tfm = np.eye(4,4)
    Tfm[0:3,0:3] = R
    Tfm[0:3,3] = t
        
    Y = Tfm.dot(np.r_[X.T, np.ones([1,4])])
    Y = Y[0:3,:].T
    print X
    print Y
    
    fTfm = find_rigid_tfm(X,Y)
    print Tfm
    print fTfm
    print np.allclose(Tfm, fTfm)
    
def test_rigidtfm2_noise():
    X = np.array([[1,0,0],[0,2,0],[0,0,3],[1,1,1]])
    R = rotation_matrix(np.array([1,2.2,3]),2.1)
    t = np.array([3,4.4,3])
    Tfm = np.eye(4,4)
    Tfm[0:3,0:3] = R
    Tfm[0:3,3] = t
    
    noise = np.random.randn(X.shape[0], X.shape[1])*0.01
    
    Y = Tfm.dot(np.r_[X.T, np.ones([1,4])])
    Y = Y[0:3,:].T + noise
    print X
    print Y

    fTfm = find_rigid_tfm(X,Y)
    print Tfm
    print fTfm
    print np.allclose(Tfm, fTfm)
#main()