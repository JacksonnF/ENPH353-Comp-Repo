#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Quaternion
import numpy as np
import time
from tensorflow import keras

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import subprocess

import roslaunch

import csv

from pynput import keyboard

collected_plates_arr = []


def hsv_pedestrian(img):
    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the color
    color1HSV = np.array([102.4,58.65,61.2])
    # color2HSV = np.array([0,0,122])
    # color3HSV = np.array([0,0,201])
    colorVariance = np.array([20,20,20])

    # Apply the HSV filter
    mask = cv2.inRange(hsv, color1HSV-colorVariance, color1HSV+colorVariance)

    # Erode the masks to remove noise
    erodeIterations = 3
    kernel1 = np.ones((2,2), np.uint8)
    eroded1 = cv2.erode(mask, kernel1, iterations=erodeIterations)
    
    blurred = cv2.GaussianBlur(eroded1, (5,5), 0)

    return blurred


def callback(data):
    global carNumber
    global current_frame
    global current_croppedLetters_individual
    global count
    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {}".format(e))

    cv2.imshow(hsv_pedestrian(np.array(cv_image)))
    cv2.waitKey(1)

    # Store current frame
bridge = CvBridge()
current_frame = None


def hsv_pedestrian(self, img):
    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the color
    color1HSV = np.array([102.4,58.65,61.2])
    # color2HSV = np.array([0,0,122])
    # color3HSV = np.array([0,0,201])
    colorVariance = np.array([20,20,20])

    # Apply the HSV filter
    mask = cv2.inRange(hsv, color1HSV-colorVariance, color1HSV+colorVariance)

    # Erode the masks to remove noise
    erodeIterations = 3
    kernel1 = np.ones((2,2), np.uint8)
    eroded1 = cv2.erode(mask, kernel1, iterations=erodeIterations)
    
    blurred = cv2.GaussianBlur(eroded1, (5,5), 0)

    return blurred


            

if __name__ == '__main__':
    global carNumber
    carNumber = 1
    global session

    time.sleep(1)
    rospy.init_node('image_isolator')
    image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback)
    licensePlateList = []




    
    
    
    rospy.spin()