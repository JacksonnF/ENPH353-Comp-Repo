#! /usr/bin/env python3
import rospy 
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import numpy as np
import time

class image_converter:

  def __init__(self):
    # self.image_pub = rospy.Publisher("image_topic_2",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',
    Image,self.callback)
    self.twist = Twist()
    self.twist.angular.z = 0.0
    self.twist.linear.x = 0.0
    self.cmd_vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)
    self.aligned = 0
    self.current_frame = []
    self.prev_frame = []
    self.stop = False
    self.look_for_ped_flag = False
    self.ped_seen_count = 0
    self.state = 0
    


  def check_crosswalk_dist(self, img):
    filtered = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(filtered, np.array([0, 50, 50]), np.array([10, 255, 255]))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    if len(contours) > 1:
      area1 = cv2.contourArea(contours[0])
      area2 = cv2.contourArea(contours[1])
      print("area1: ", area1)
      print("area2: ", area2)
      


  def align_robot(self, img):
    min_line_length = 100
    max_line_gap = 80
    rho = 1
    theta = np.pi / 180
    threshold = 185

    filtered = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(filtered, np.array([0, 50, 50]), 
                       np.array([10, 255, 255]))   
    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), 
                            min_line_length, max_line_gap)
    
    if lines is not None:
        x1, y1, x2, y2 = lines[0][0]
        deg = np.rad2deg(np.arctan((y2-y1)/(x2-x1)))
        # print(deg)
        if self.aligned == 0 or self.aligned == 1:
            if deg > 3:
                self.twist.angular.z = -0.5
            elif deg < -3:
                self.twist.angular.z = 0.5
            else:
                self.twist.angular.z = 0.0
                print("aligned")
                print(self.aligned)
                self.aligned += 1
        
    if self.aligned == 2:
       self.state = 2
    self.cmd_vel_pub.publish(self.twist)

  def look_for_ped(self):
     print(np.mean(cv2.absdiff(self.current_frame, self.prev_frame))**2)
     if (abs(np.mean(cv2.absdiff(self.current_frame, self.prev_frame))**2) > 0.06):
        print('go')
        self.ped_seen_count += 1
        if self.ped_seen_count > 2:
            self.twist.linear.x = 0.3
            self.cmd_vel_pub(self.twist)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    min_line_length = 100
    max_line_gap = 80
    rho = 1
    theta = np.pi / 180
    threshold = 185


    w = cv_image.shape[1]
    h = cv_image.shape[0]

    cropped_img = cv_image[h-240:h, 0:w]

    blur = cv2.GaussianBlur(cropped_img, (5, 5), 0)
    filtered = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(filtered, np.array([0, 0, 75]), np.array([5, 5, 90]))
    edges = cv2.Canny(mask, 50, 150)

    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]), 
                            min_line_length, max_line_gap)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area1 = cv2.contourArea(contours[0])
    print(area1)
    #draw contours on cropped image
    cv2.drawContours(cropped_img, contours, -1, (0, 255, 0), 3)
    if lines is not None:
      for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2-y1)/(x2-x1)
        # if abs(slope) < 0.5:
          # cv2.line(cropped_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("image", cropped_img)
    cv2.waitKey(3)
    


def main():
    # rospy.init_node('topic_publisher')
    # pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.init_node('image_converter', anonymous=True)
    ic = image_converter()
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