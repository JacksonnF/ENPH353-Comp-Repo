#! /usr/bin/env python3
import rospy 
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
import numpy as np
import time
import conv_net
import torch

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
    
    self.params = cv2.SimpleBlobDetector_Params()
    self.params.filterByArea = True
    self.params.minArea = 8
    self.params.maxArea = 100
    self.detector = cv2.SimpleBlobDetector_create(self.params)

    # Initialize the previous centroid and timestamp
    self.prev_centroid = None
    self.prev_time = None



  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    w = cv_image.shape[1]
    h = cv_image.shape[0]
    cropped_img = cv_image[h-240:h, 0:w]
    blur = cv2.GaussianBlur(cropped_img, (5, 5), 0)
    filtered = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(filtered, np.array([13, 72, 91]), np.array([37, 166, 183]))
    area = cv2.countNonZero(mask)
    print(area)
    cv2.imshow("Image window", mask)
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
