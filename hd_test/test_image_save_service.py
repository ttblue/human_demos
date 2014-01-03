import rospy
import time
import roslib
roslib.load_manifest('ar_track_service')

from ar_track_service.srv import SaveImage, SaveImageRequest, SaveImageResponse
 
rospy.init_node("test_image_save_service")

save_service = rospy.ServiceProxy("saveImages", SaveImage)

req = SaveImageRequest ()
req.start = True
req.folder_name = "/home/sibi/temp/check_service"

save_service(req)

time.sleep(2)

req.start = False
save_service(req)
