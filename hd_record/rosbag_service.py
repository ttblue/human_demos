from threading import Thread
import rospy, rosbag

from hd_utils.colorize import *

class TopicWriter (Thread):
    """
    Class which stores messages from a list of topics when needed.
    """
    def __init__(self, topics, topic_types):
        Thread.__init__(self)

        if rospy.get_name() == '/unnamed':
            rospy.init_node('bag_writer')
        
        self.topics = topics
        self.saving = False
        # Make sure that saving during recording and saving right after recording do not collide
        self.lock = False
        self.bag = None
        self.done = False
        
        self.topic_lists = {}
        self.counters = {}
        self.topic_subs = {}
        for topic,topic_type in zip(topics,topic_types):
            self.topic_lists[topic] = []
            self.counters[topic] = 0
            self.topic_subs[topic] = \
            rospy.Subscriber(topic, topic_type, callback=self.callback, callback_args=topic)
        
        self.start()

     
    def callback (self, msg, topic):
        """
        Stores messages to specific topic.
        """
        ts = rospy.Time.now()
        if self.saving:
            if msg._has_header:
                ts = msg.header.stamp
            self.topic_lists[topic].append((msg,ts))
    
    def add_topics (self, topics, topic_types):
        """
        Add elements from topics into topics of class.
        """
        if isinstance(topics, list):
            for topic,topic_type in zip(topics,topic_types):
                if topic not in self.topics:
                    self.topics.append(topic)
                    self.topic_lists[topic] = []
                    self.counters[topic] = 0
                    self.topic_subs[topic] = \
                    rospy.Subscriber(topic, topic_type, callback=self.callback, callback_args=topic)
        else:
            if topics not in self.topics:
                self.topics.append(topics)
                self.topic_lists[topics] = []
                self.counters[topics] = 0
                self.topic_subs[topics] = \
                rospy.Subscriber(topics, topic_types, callback=self.callback, callback_args=topics)
    
    def start_saving (self, file):
        """
        Saves to specific file or stream.
        """
        if not self.saving:
            blueprint("Saving to file %s."%file)
            self.bag = rosbag.Bag(file, mode='w')
            self.reset_lists() #Just in case
            self.saving = True
            self.lock = True

    
    def run (self):
        """
        Saves topics in a separate thread.
        """
        rate = rospy.Rate(30)
        while True and not self.done:
            while self.saving: # Start saving WHILE recording
                for topic in self.topics:
                    if self.counters[topic] < len(self.topic_lists[topic]):
                        msg, ts = self.topic_lists[topic][self.counters[topic]]
                        self.bag.write(topic, msg, ts)
                        self.counters[topic] += 1
            self.lock = False
            rate.sleep()
        
        blueprint("Topic Writer thread has finished.")
    
    def stop_saving (self):
        """
        Stop saving to the topics.
        """
        if self.saving:
            self.saving = False
            
            # Wait until lock aquired: wait until saving thread stops
            rate = rospy.Rate(30)
            while self.lock:
                rate.sleep()

            for topic in self.topics:
                while self.counters[topic] < len(self.topic_lists[topic]):
                    msg, ts = self.topic_lists[self.counters[topic]]
                    self.bag.write(topic, msg, ts)
                    self.counters[topic] += 1
            
            self.bag.close()
            self.reset_lists()
            blueprint("Saved bagfile.")
            
    
    def reset_lists (self):
        """
        Reset the lists.
        """
        for topic in self.topic_lists:
            self.topic_lists[topic] = []
            self.counters[topic] = 0
    
    def done (self):
        """
        Done completely.
        """
        self.done = True