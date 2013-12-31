import rospy
from sensor_msgs.msg import Image

rospy.init_node('check_resolution')

val = None
check = False

def cb (msg):
    global val,check
    if not check:
        val = msg
        check = True

sub = rospy.Subscriber('/camera1/image_raw', Image, callback=cb)

rate = rospy.Rate(1)

while not check:
    rate.sleep()

print "Width:",val.width,"\nHeight:",val.height
