#!/usr/bin/python
import os
import time
import subprocess
import rospy, roslib
import argparse
from std_msgs.msg import Float32

devnull = open(os.devnull, 'wb')

left_state = ''
right_state = ''
first_left = True
first_right = True

latest_left_angle = -1
latest_right_angle = -1

threshold = 15

def call_back_left(msg):
    global latest_left_angle, first_left, left_state, threshold
    if first_left:
        first_left = False
        if msg.data > threshold:
            left_state = 'open'
        else:
            left_state = 'closed'
    latest_left_angle = msg.data

def call_back_right(msg):
    global latest_right_angle, first_right, right_state, threshold
    if first_right:
        first_right = False
        if msg.data > threshold:
            right_state = 'open'
        else:
            right_state = 'closed'
    latest_right_angle = msg.data
    
def update_states(left_angle, right_angle):
    global threshold, left_state, right_state
    if left_angle > threshold:
        left_state = 'open'
    else:
        left_state = 'closed'
    if right_angle > threshold:
        right_state = 'open'
    else:
        right_state = 'closed'

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--threshold", help="Threshold pot angle for distinguishing gripper open and closed states", default=15, type=int)

    args = parser.parse_args()
    
    threshold = args.threshold
    
    rospy.init_node('gripper_feedback')

    sub_left = rospy.Subscriber('/l_pot_angle', Float32, call_back_left)
    sub_right = rospy.Subscriber('/r_pot_angle', Float32, call_back_right)
    
    
    
    while (first_left or first_right) and not rospy.is_shutdown():
        time.sleep(0.033)

    if not rospy.is_shutdown():
        print "First pot angles received."

    while not rospy.is_shutdown():
        # Store local copies of angle readings so they do not change
        left_angle = latest_left_angle
        right_angle = latest_right_angle
        
        # Both Open
        if left_state == 'closed' and right_state == 'closed' and right_angle > threshold and left_angle > threshold:
            update_states(left_angle, right_angle)
            subprocess.call("espeak -v en 'Both opened.'", stdout=devnull, stderr=devnull, shell=True)
            continue
        
        # Both Closed
        elif left_state == 'open' and right_state == 'open' and right_angle < threshold and left_angle < threshold:
            update_states(left_angle, right_angle)
            subprocess.call("espeak -v en 'Both closed.'", stdout=devnull, stderr=devnull, shell=True)
            continue
        # Left Open
        if left_state == 'closed' and left_angle > threshold:
            subprocess.call("espeak -v en 'Left Opened.'", stdout=devnull, stderr=devnull, shell=True)
        # Right Open
        if right_state == 'closed' and right_angle > threshold:
            subprocess.call("espeak -v en 'Right Opened.'", stdout=devnull, stderr=devnull, shell=True)
        # Left Close
        if left_state == 'open' and left_angle < threshold:
            subprocess.call("espeak -v en 'Left Closed.'", stdout=devnull, stderr=devnull, shell=True)
        # Right Close
        if right_state == 'open' and right_angle < threshold:
            subprocess.call("espeak -v en 'Right Closed.'", stdout=devnull, stderr=devnull, shell=True)
        
        update_states(left_angle, right_angle)
        # Delay
        time.sleep(0.3)
