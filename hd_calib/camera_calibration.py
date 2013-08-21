import rospy
import numpy as np
import cv2

from hd_utils import ros_utils as ru, clouds, conversions
from get_transform import find_rigid_tfm
import checkerboard_transform as cbt

asus_xtion_pro_f = 544.260779961

def get_ar_marker_poses (rgb, depth):
    """
    In order to run this, ar_marker_service needs to be running.
    """
    if rospy.get_name() == '/unnamed':
        rospy.init_node('keypoints')
    
    getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)
    
    #xyz = svi.transform_pointclouds(depth, tfm)
    xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
    pc = ru.xyzrgb2pc(xyz, rgb, '/base_footprint')
    
    req = MarkerPositionsRequest()
    req.pc = pc
    
    marker_tfm = {}
    res = getMarkers(req)
    for marker in res.markers.markers:
        marker_tfm[marker.id] = conversions.pose_to_hmat(marker.pose.pose)
    
    #print "Marker ids found: ", marker_tfm.keys()
    
    return marker_tfm
   
def find_common_ar_markers (ar_pos1, ar_pos2):
    """
    Finds the common markers between two sets of markers.
    """
    ar1, ar2 = {}, {}
    common_id = np.intersect1d(ar_pos1.keys(), ar_pos2.keys())
    
    for i in common_id:
        ar1[i] = ar_pos1[i]
        ar2[i] = ar_pos2[i]
        
    return ar1, ar2

def convert_hmats_to_points (hmats):
    """
    Adds a point for the origin and one for each axis.
    Could be readily changed for something similar.
    """
    
    dist = 0.05
    points = []
    for hmat in hmats:
        x,y,z,p = hmat[0:3].T
        
        points.append(p)
        points.append(p+dist*x)
        points.append(p+dist*y)
        points.append(p+dist*z)
        
    return points


def get_click_points (rgb, depth):
    """
    Get clicked points in an image.
    Add functions if you want to calibrate based on color or something.
    """
    clicks = []

    def mouse_click(event, x, y, flags, params):
        if event == cv.CV_EVENT_LBUTTONDOWN:
            clicks.append([x, y])

    xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
            
    cv2.imshow("Points", rgb)
    cv.SetMouseCallback("Points", mouse_click, 0)
    print "Click the points keeping the order in mind. Press enter to continue."
    cv2.waitKey()
    cv2.destroyWindow("Points")
    
    clickPoints = []

    for (x, y) in clicks:
        clickPoints.append(np.copy(xyz[y][x]))
    
    return clickPoints


########################## HELPER FUNCTIONS ABOVE ################################
########################## CALIBRATION FUNCTIONS BELOW ###########################

        
def calibrate_ar (rgb1, depth1, rgb2, depth2):
    """
    Create points for the AR transform and then find a transform such that
    Tfm * cam1 = cam2 
    """
    
    ar_pos1 = get_ar_marker_poses (rgb1, depth1)
    ar_pos2 = get_ar_marker_poses (rgb2, depth2)
    
    points1 = conver_hmat_to_points(ar_pos1)
    points2 = conver_hmat_to_points(ar_pos2)
    
    return find_rigid_tfm(points1, points2)


cb_rows = None
cb_cols = None
def calibrate_cb (rgb1, depth1, rgb2, depth2):
    """
    Find CB transform for each cam and then return a transform such that
    Tfm * cam1 = cam2.
    TODO: Find better method to get transform out of checkerboard.
    """

    rtn1, corners1 = cbt.get_corners_rgb(rgb1, cb_rows, cb_cols)
    xyz1 = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
    rtn2, corners2 = cbt.get_corners_rgb(rgb2, cb_rows, cb_cols)
    xyz2 = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
    
    if rtn1 != rtn2:
        print "Warning: One of the cameras do not see all the checkerboard points."
    elif rtn1 == 0:
        print "Warning: Not all the checkerboard points are seen by the cameras."
        
    if len(corners1) != len(corners2):
        print "Warning: Different number of points found by different cameras -- transform will be off."
    
    tfm1 = cbt.get_svd_tfm_from_points(cbt.get_xyz_from_corners(corners1, xyz1))
    tfm2 = cbt.get_svd_tfm_from_points(cbt.get_xyz_from_corners(corners2, xyz2))
    
    return np.linalg.inv(tfm1).dot(tfm2)


def calibrate_click (rgb1, depth1, rgb2, depth2):
    """
    Uses points clicked for each cam and then return a transform such that
    Tfm * cam1 = cam2.
    TODO: Maybe use red color or something to tack clicked point onto?
    """
    
    points1 = get_click_points(rgb1, depth1)
    points2 = get_click_points(rgb2, depth2)
     
     
     
    return find_rigid_tfm(points1, points2)