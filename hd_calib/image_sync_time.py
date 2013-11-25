import rospy
from sensor_msgs.msg import Image


class image_time_sync:
    
    def __init__(self, topic1, topic2):
        
        rospy.init_node('img_time_sync_node')


        self.image1 = None
        self.image2 = None
        self.pending1 = False
        self.pending2 = False
        
        self.finished = False
        self.img_sub1 = rospy.Subscriber(topic1, Image, callback=self.image1_cb)
        self.img_sub2 = rospy.Subscriber(topic2, Image, callback=self.image2_cb)        
        self.img_pub1 = rospy.Publisher('/camera1_image', Image)
        self.img_pub2 = rospy.Publisher('/camera2_image', Image)
        
        self.run()
        
        
    
    def image1_cb (self, img):
        self.image1 = img
        self.pending1 = True
        
    def image2_cb (self, img):
        self.image2 = img
        self.pending2 = True
        
    def run (self):
        sleeper = rospy.Rate(30)
        while not self.finished:
            if self.pending1 and self.pending2:
                ts = rospy.Time.now()
                self.image1.header.stamp = ts
                self.image2.header.stamp = ts
                self.img_pub1.publish(self.image1)
                self.img_pub2.publish(self.image2)
                self.pending1 = False 
                self.pending2 = False
            sleeper.sleep()
            
        
    def set_finished (self, finished=True):
        self.finished = finished
        
if __name__=='__main__':
    image_time_sync('/camera1/rgb/image_raw', '/camera2/rgb/image_raw')