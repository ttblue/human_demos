import rospy

import roslib
roslib.load_manifest('cv_bridge')
roslib.load_manifest('ar_track_service')

from ar_track_service.srv import MarkerImagePositions, MarkerImagePositionsRequest, MarkerImagePositionsResponse 

import cv, cv2
from cv_bridge import CvBridge

br = CvBridge()

rospy.init_node("test_image_service")

mPos = rospy.ServiceProxy("getImageMarkers    ", MarkerImagePositions)

req = MarkerImagePositionsRequest ()
req.img = br.cv_to_imgmsg(cv.LoadImage("/home/sibi/Downloads/artags.png"))

res = mPos(req)