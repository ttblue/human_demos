import rospy
import cv, cv2
import numpy as np

from sensor_msg.msg import PointCloud2

from hd_utils import ros_utils as ru

def get_corners_rgb(rgb,rows=6,cols=8):
    cv_rgb = cv.fromarray(rgb)
    
    rtn, corners = cv.FindChessboardCorners(cv_rgb, (cb_rows, cb_cols))
    return rtn, corners

def get_xyz_from_corners (corners, xyz):
    points = []
    for j,i in corners:
        x = i - np.floor(i)
        y = j - np.floor(j)
        p1 = xyz[np.floor(i),np.floor(j)]
        p2 = xyz[np.floor(i),np.ceil(j)]
        p3 = xyz[np.ceil(i),np.ceil(j)]
        p4 = xyz[np.ceil(i),np.floor(j)]        
        p = p1*(1-x)*(1-y) + p2*(1-x)*y + p3*x*y + p4*x*(1-y)
        if np.isnan(p).any(): print p
        points.append(p)

    return np.asarray(points)

def get_corners_from_pc(pc,rows=6,cols=8):
    xyz, rgb = ru.pc2xyzrgb(pc)
    rgb = np.copy(rgb)
    rtn, corners = get_corners_rgb(rgb, rows, cols)
    if len(corners) == 0:
        return 0, None
    points = get_xyz_from_corners(corners, xyz)
    return rtn, points

class chessboardPC:
    """
    Class to show images and find chessboard corners.
    """
    def __init__(self, topic):
        self.topic = topic
        self.row_cb = 6
        self.col_cb = 8

        self.latest_pc = None
        self.latest_img = None

        if rospy.get_name() == '/unnamed':
            rospy.init_node('test_cb')
        
        self.pc_sub = rospy.Subscriber(topic,PointCloud2,call_back=self.pc_callback)

    def image_callback(self, msg):
        try:
            self.latest_img = self.bridge.imgmsg_to_cv(data, "bgr8")
        except CvBridgeError, e:
            print e
        
    def pc_callback(self, msg):
        self.latest_pc = msg
        _, self.latest_img = ru.pc2xyzrgb(msg)
        self.latest_img = np.copy(self.latest_img)
        
    def get_cb_corners(self, method='cv'):
        if method=='cv':
            rtn, corners = get_corners_from_pc
            return corners
        else:
            return None