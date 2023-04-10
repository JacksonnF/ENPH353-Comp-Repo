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

        # Apply a color threshold to the image to detect the pedestrian
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array([92,106,43]), np.array([106,151,52]))
    # cv2.imshow('Pedestrian Tracker', mask)
    # cv2.waitKey(1)

    # Detect the blobs in the binary image
    keypoints = self.detector.detect(mask)
    # print(keypoints)

    # If at least one blob is detected, track the pedestrian
    if len(keypoints) > 0:
        # Get the centroid of the first blob
        centroid = keypoints[0].pt

        # Draw a circle around the pedestrian
        cv2.circle(cv_image, (int(centroid[0]), int(centroid[1])), int(keypoints[0].size/2), (0, 255, 0), 2)

        # If this is the first frame or the previous centroid is not available, set the previous centroid to the current centroid and the previous time to the current time
        if self.prev_centroid is None:
            self.prev_centroid = centroid
            self.prev_time = time.time()
        else:
            # Calculate the distance between the current centroid and the previous centroid
            dist = np.sqrt((centroid[0]-self.prev_centroid[0])**2 + (centroid[1]-self.prev_centroid[1])**2)

            # Calculate the time difference between the current frame and the previous frame
            time_diff = time.time() - self.prev_time

            # Calculate the velocity of the pedestrian
            velocity = dist / time_diff

            # Publish the velocity to a ROS topic
            print(velocity)

            # Set the previous centroid to the current centroid and the previous time to the current time
            self.prev_centroid = centroid
            self.prev_time = time.time()

    # Display the image with the pedestrian and the velocity estimation
 
  


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
