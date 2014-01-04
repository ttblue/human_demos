import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PointStamped

from hd_record.rosbag_service import TopicWriter

rospy.init_node('rb_service_test')

ipub = rospy.Publisher('a', Int32)
ppub = rospy.Publisher('b', PointStamped)

tw = TopicWriter(topics=['/a','/b'], topic_types=[Int32, PointStamped])

filename = '/home/sibi/temp/bag_test/bag.bag'

tw.start_saving(filename)
print 'now'

rate = rospy.Rate(10)
for i in range(10):
    print i
    ipub.publish(i)
    p = PointStamped()
    p.header.stamp = rospy.Time.now()
    p.point.x = i
    ppub.publish(p)
    rate.sleep()
    
tw.stop_saving()
print 'end'