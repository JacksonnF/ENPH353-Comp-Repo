#! /usr/bin/env python3
import rospy 
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pynput import keyboard
import time
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import plate_detect as pd
import math

end_global = False


class image_converter:

    def __init__(self):
        # self.image_pub = rospy.Publisher("image_topic_2",Image)

        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',
        Image,self.callback)
        self.cmd_vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
        self.license_plate_pub = rospy.Publisher('/license_plate', String, queue_size=1)
        self.twist = Twist()
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        

        self.start = True
        self.end = False

        self.finished_manuever = False
        self.pred_count = 0
        self.ended_flag = 0
        self.num_of_crosswalks = 0
        self.ped_routine = False
        self.previous_area = 0
        self.aligned = 0
        self.see_cross_twice = 0
        self.waiting = 0


        self.waiting_to_cross = False
        self.crossing = False
        self.prev_img = []
        self.i = 0
        self.stop_at_crosswalk = False
        self.area_seen_twice = 0
        self.start_time = time.time()
        self.been_on_sand = 0
        self.loop_count = 0
        self.enter_inner_loop = False
        self.align_cross_inner = False
        self.inner_manuever = False
        self.fucked = False
        self.get_to_sand = False

        self.stop_for_truck = False

        self.numberOfPredictions = 0

        self.continueThroughCrosswalk = False
        self.approachingCrosswalk = False
        self.pedestrianMiddleTime = None


        # Stuff for pedestrian routine
        self.lastCentroid = (None,None)
        self.lastPedestrianSpeed = None
    
    
    def hsvCrosswalk(self,img):
        filtered = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(filtered, np.array([0, 52, 128]), np.array([0, 255, 255]))
        mask2 = cv2.inRange(filtered, np.array([0, 3, 116]), np.array([16, 255, 255]))
        eroded = cv2.erode(mask2, np.ones((2, 2), np.uint8), iterations=1)
        dilated = cv2.dilate(eroded, np.ones((3, 3), np.uint8), iterations=3)
        dilateHorizontally = cv2.dilate(dilated, np.ones((1, 3), np.uint8), iterations=10)
        erodeHorizontally = cv2.erode(dilateHorizontally, np.ones((1, 3), np.uint8), iterations=10)
        return erodeHorizontally


    def check_crosswalk_dist(self, img):
        # filtered = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(filtered, np.array([0, 52, 128]), np.array([0, 255, 255]))
        # eroded = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)
        # dilated = cv2.dilate(eroded, np.ones((3, 3), np.uint8), iterations=3)

        crosswalk = self.hsvCrosswalk(img)
        contours, hierarchy = cv2.findContours(crosswalk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        if len(contours) > 1:
            area1 = cv2.contourArea(contours[0])
            print(area1)
            area2 = cv2.contourArea(contours[1])
            # print(area1, area2)
            return (area1, area2)
        else:
            return (0, 0)
    
    def check_if_approaching_crosswalk(self, img):
        area1, area2 = self.check_crosswalk_dist(img)
        print(area1)
        if(100<area1<2200):
                self.approachingCrosswalk = True
        else:
                self.approachingCrosswalk = False
            
    def check_pedestrian_walking(self, img):
        pedestrian = self.hsv_pedestrian(img)
        pedestrianContours = self.contour_pedestrian(pedestrian)
        pedestrianX, pedestrianY = self.find_contour_centroid(pedestrianContours[0])
        crossWalkXmin, crossWalkXmax = self.find_crosswalk_x_extrema(img)
        if(crossWalkXmin < pedestrianX < crossWalkXmax):
            self.continueThroughCrosswalk = True
        return
    
    def check_pedestrian_walking2(self, img):
        pedestrian = self.hsv_pedestrian(img)
        pedestrianContours = self.contour_pedestrian(pedestrian)
        pedestrianX, pedestrianY = self.find_contour_centroid(pedestrianContours[0])
        crossWalkXmin, crossWalkXmax = self.find_crosswalk_x_extrema(img)
        averageCrossWalkX = (crossWalkXmin + crossWalkXmax)/2
        if(averageCrossWalkX-3<pedestrianX<averageCrossWalkX+3):
            self.pedestrianMiddleTime = time.time()
            print("Pedestrian time initialized")
        return

    def find_crosswalk_x_extrema(self, img):
        crosswalk = self.hsvCrosswalk(img)
        contours, hierarchy = cv2.findContours(crosswalk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        if len(contours) > 1:
            contour_points = contours[0].reshape((-1, 2))  # Flatten the contour array
            left_x = np.min(contour_points[:, 0])  # Find the minimum x value (furthest left)
            right_x = np.max(contour_points[:, 0])  # Find the maximum x value (furthest right)
            return left_x, right_x
        else:
            return 0, 0
            
        
    
    def hsv_pedestrian(self, img):
        # Convert BGR to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Apply the HSV filter
        mask = cv2.inRange(hsv, np.array([92,106,43]), np.array([106,151,52]))

            # Erode the masks to remove noise
        dilateIterations3 = 5
        kernel3 = np.ones((2,2), np.uint8)
        dilated3 = cv2.dilate(mask, kernel3, iterations=dilateIterations3)


        dilateIterations2 = 5
        kernel2 = np.ones((1,1), np.uint8)
        dilated2 = cv2.dilate(dilated3, kernel2, iterations=dilateIterations2)

        blurred = cv2.GaussianBlur(dilated2, (5,5), 0)

        return blurred

    
    def contour_pedestrian(self, img):
        # Find the contours in the thresholded image
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Aproximate the contours
        approx_contours = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
            approx_contours.append(approx)

        if(len(approx_contours)>0):
            currentMaxArea = 0
            index = 0
            for i in range(0,len(approx_contours)):
                currentArea = cv2.contourArea(approx_contours[i])
                if(currentArea>currentMaxArea):
                    index = i
                    currentMaxArea = currentArea

        return [approx_contours[index]]


    def find_contour_centroid(self, contour):
        M = cv2.moments(contour)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return [cx, cy]


    def callback(self,data):


        cv2.waitKey(1)
        if self.start:
            self.start = False
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.prev_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            self.start_time = time.time()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        #-----------------------------------

        # areas = self.check_crosswalk_dist(cv_image)
        # self.check_if_approaching_crosswalk(areas[0])
        # if(self.approachingCrosswalk and self.pedestrianMiddleTime == None):
        #     print('approaching crosswalk')
        #     self.check_pedestrian_walking2(cv_image)

        # if(self.pedestrianMiddleTime != None):
        #     modulus = math.fmod(time.time()-self.pedestrianMiddleTime,4)
        #     if(modulus<0.05 or modulus >3.95):
        #         print("Pedestrian Center")
        
        pedestrianHsvImage = self.hsv_pedestrian(cv_image)
        crossWalkHsvImage = self.hsvCrosswalk(cv_image)
        # crossWalkContours = self.check_crosswalk_dist(cv_image)
        # print("Area0: " + str(crossWalkContours[0]) + " Area1: " + str(crossWalkContours[1]))


        color_img = np.zeros((720, 1280, 3), np.uint8)

        color_img[:,:,0] = pedestrianHsvImage # Red channel
        color_img[:,:,1] = crossWalkHsvImage # Green channel

        cv2.imshow("pedestrianHsv", color_img)
        cv2.waitKey(1)
        #-----------------------------------

        self.check_if_approaching_crosswalk(cv_image)
        if(self.approachingCrosswalk):
            print("Approaching crosswalk")
            self.check_pedestrian_walking(cv_image)
            if(self.continueThroughCrosswalk):
                print("Continue through crosswalk")
            else:
                print("Stop at crosswalk")

def main():
    # rospy.init_node('topic_publisher')
    # pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()

    
    # while not rospy.is_shutdown():
    #     i = 1
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    # image_sub = rospy.Subscriber('/rrbot/camera1/image_raw', Image, )

    # rate = rospy.Rate(2)
    # move = Twist()
    # move.linear.x = 0.5
    # move.angular.z = 0.5

    # while not rospy.is_shutdown():
    #     pub.publish(move)
    #     rate.sleep()