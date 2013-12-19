#!/usr/bin/python
from visualize_phasespace import publish_phasespace_markers_ros
from visualize_ar import visualize_ar

from threading import Thread
import subprocess
import rospy


class threadClass (Thread):
    def run (self):
        publish_phasespace_markers_ros()
    

subprocess.call("killall XnSensorServer", shell=True)

rospy.init_node("visualize_ps_ar")

thc = threadClass ()
thc.start()

visualize_ar()