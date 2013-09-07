import rospy
from sensor_msgs.msg import PointCloud2

pc = None
def cb (msg):
    global pc
    rospy.loginfo('I got point cloud.')
    pc = msg

rospy.init_node('stuff')
ps_sub = rospy.Subscriber('camera1/depth_registered/points',PointCloud2, callback=cb)
