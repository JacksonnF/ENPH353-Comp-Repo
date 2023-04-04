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
    

    listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
    listener.start()
    
  def spawn_position(self, x, y, z, o_x=0, o_y=0, o_z=0, o_w=0):

    msg = ModelState()
    msg.model_name = 'R1'

    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.position.z = z
    msg.pose.orientation.x = o_x
    msg.pose.orientation.y = o_y
    msg.pose.orientation.z = o_z
    msg.pose.orientation.w = o_w

    rospy.wait_for_service('/gazebo/set_model_state')
    try:
        set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        resp = set_state( msg )

    except rospy.ServiceException:
        print ("Service call failed")

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
        print(deg)
        if deg > 1:
            self.twist.linear.x = 0.0
            self.twist.angular.z = -1.1
        elif deg < -1:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 1.1
        else:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.0
            print("aligned")
            self.aligned += 1
        self.cmd_vel_pub.publish(self.twist)
        if self.aligned > 2:
          self.waiting_to_cross = True
          self.stop_at_crosswalk = False
          self.aligned = 0
          self.num_of_crosswalks += 1
          #  self.predict_sand(bottom_half_rgb)
        
    if lines is None:
      print('no lines found')
      self.waiting += 1
      if self.waiting > 3:
        self.waiting_to_cross = True
        self.stop_at_crosswalk = False
        self.aligned = 0
        self.num_of_crosswalks += 1
        self.waiting = 0
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        #  self.predict_sand(bottom_half_rgb)

  def check_crosswalk_dist(self, img):
    filtered = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(filtered, np.array([0, 50, 50]), np.array([10, 255, 255]))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    if len(contours) > 1:
      area1 = cv2.contourArea(contours[0])
      area2 = cv2.contourArea(contours[1])
      # print(area1, area2)
      return (area1, area2)
    else:
      return (0, 0)


  def predict(self, img):
    img_aug = np.expand_dims(img, axis=0)
    input_data = np.expand_dims(np.array(img_aug, dtype=np.float32), axis=-1)
    # input_data = np.array(img_aug, dtype=np.float32)
    interpreter = tf.lite.Interpreter(model_path="/home/fizzer/ros_ws/src/controller_pkg/data/model_3000c_quantized.tflite")
    interpreter.allocate_tensors()

    # Set the input tensor.
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_data)

    # Run inference.
    interpreter.invoke()

  # Get the output tensor.
    output_index = interpreter.get_output_details()[0]["index"]
    output_data = interpreter.get_tensor(output_index)

    # Print the predicted class.
    predicted_class = np.argmax(output_data)
    # print(predicted_class)
    return output_data
  
  def predict_sand(self, img):
    img_aug = np.expand_dims(img, axis=0)
    # input_data = np.expand_dims(np.array(img_aug, dtype=np.float32), axis=-1)
    input_data = np.array(img_aug, dtype=np.float32)
    interpreter = tf.lite.Interpreter(model_path="/home/fizzer/ros_ws/src/controller_pkg/data/model_100_quantized.tflite")
    interpreter.allocate_tensors()

    # Set the input tensor.
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_data)

    # Run inference.
    interpreter.invoke()

  # Get the output tensor.
    output_index = interpreter.get_output_details()[0]["index"]
    output_data = interpreter.get_tensor(output_index)

    # Print the predicted class.
    predicted_class = np.argmax(output_data)
    # print(predicted_class)
    return output_data
  
  def crop_for_prediction(self, img, type):
    scale_percent = 20
    w = int(img.shape[1] * scale_percent / 100)
    h = int(img.shape[0] * scale_percent / 100)
    dim = (w, h)
    if type: #sand
      resized_img_rgb = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
      bottom_half_rgb = resized_img_rgb[int(h/2):h, 0:w]
      return bottom_half_rgb
    else: #binary thresholded
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      resized_img_gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
      ret, bin_img = cv2.threshold(resized_img_gray[int(h/2):h, 0:w], 180,255,0)
      return bin_img

  
  def callback(self,data):
    if self.start:
      self.license_plate_pub.publish(str('TeamRed,multi12,0,XR58'))
      self.spawn_position(-0.85, 0 , 0.5, 0,0,1,0)
      self.initial_turn()
      self.start = False
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      self.prev_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    if self.stop_at_crosswalk:
      self.align_robot(cv_image)
      return
       
    if self.waiting_to_cross:
      print('waiting to cross')
      #  gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
      #  print(np.mean(self.prev_img - gray)**2)
      #  self.prev_img = gray
      self.i += 1
      if self.i > 25:
        self.waiting_to_cross = False
        self.crossing = True
        self.i = 0
    
      return
    
    #drive forward until both crosswalks are out of view
    if self.crossing:
      print('crossing')
      contours = self.check_crosswalk_dist(cv_image)
      if contours[0] < 150 and contours[1] < 150:
        self.crossing = False
      else:
        self.twist.linear.x = 0.3
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
      return

    
    areas = self.check_crosswalk_dist(cv_image)
    if len(areas) > 1:
      if areas[0] > 5000 and areas[1] > 200 and time.time() - self.start_time > 5:
        # self.align_robot(cv_image)
        
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        self.stop_at_crosswalk = True
        self.area_seen_twice = 0
        # self.waiting_to_cross = True
        return

    
    if self.num_of_crosswalks > 1:
      bottom_half_rgb = self.crop_for_prediction(cv_image, True)
      pred_arr = self.predict_sand(bottom_half_rgb)
    else:
      bin_img = self.crop_for_prediction(cv_image, False)
      pred_arr = self.predict(bin_img)
    pred = np.argmax(pred_arr)
    if self.num_of_crosswalks > 1:
      if (pred == 0):
        self.twist.linear.x = 0.0
        self.twist.angular.z = 1.1
      elif (pred == 1):
        self.twist.linear.x = 0.365
        self.twist.angular.z = 0.0
      elif (pred == 2):
        self.twist.linear.x = 0.0
        self.twist.angular.z = -1.1

      self.cmd_vel_pub.publish(self.twist)
    else:
      next_prob = pred_arr[0][1]
      p_scaled = min(-0.125/(next_prob - 1.0001), 1.0)
      prev_speed = self.twist.linear.x
      if (pred == 0):
        self.twist.linear.x = 0.09
        self.twist.angular.z = 1.3 #1.3 good before
      elif (pred == 1):
        self.twist.linear.x = min(prev_speed + 0.1, p_scaled) #0.6 normally
        self.twist.angular.z = 0.0
      elif (pred == 2):
        self.twist.linear.x = 0.09
        self.twist.angular.z = -1.3

      self.cmd_vel_pub.publish(self.twist)    


    # if self.ped_routine:
    #   try:
    #       cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #   except CvBridgeError as e:
    #       print(e)

    #   self.align_robot(cv_image)

    # else:
    #   global end_global
    #   if end_global:
        
    #     self.twist.linear.x = 0.0
    #     self.twist.angular.z = 0.0
    #     self.cmd_vel_pub.publish(self.twist)
    #     end_global = False
    #     self.start = True
    #   if self.start:
    #     self.license_plate_pub.publish(str('TeamRed,multi12,0,XR58'))
    #     self.spawn_position(-0.85, 0 , 0.5, 0,0,1,0)
    #     self.initial_turn()
    #     self.finished_manuever = True
    #     self.start = False
    #   if self.end and self.ended_flag == 0:
    #     self.license_plate_pub.publish(str('TeamRed,multi12,-1,XR58'))
    #     self.twist.linear.x = 0.0
    #     self.twist.angular.z = 0.0
    #     self.cmd_vel_pub.publish(self.twist)
    #     self.ended_flag = 1
        
    #   if self.finished_manuever and not self.end:
    #     try:
    #       cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #     except CvBridgeError as e:
    #       print(e)

    #     # gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    #     # scale_percent = 20
    #     # w = int(gray.shape[1] * scale_percent / 100)
    #     # h = int(gray.shape[0] * scale_percent / 100)
    #     # dim = (w, h)

    #     # resized_img_rgb = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)
    #     # bottom_half_rgb = resized_img_rgb[int(h/2):h, 0:w]

    #     # resized_img_gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)
    #     # ret, bin_img = cv2.threshold(resized_img_gray[int(h/2):h, 0:w], 180,255,0)

    #     if self.num_of_crosswalks > 1:
    #       bottom_half_rgb = self.crop_for_prediction(cv_image, True)
    #       print('predicting sand')
    #       pred_arr = self.predict_sand(bottom_half_rgb)
    #     else:
    #       bin_img = self.crop_for_prediction(cv_image, False)
    #       self.check_crosswalk_dist(cv_image)
    #       pred_arr = self.predict(bin_img)
        
        
    #     # self.pred_count += 1
    #     # if self.pred_count == 375:
    #     #   # self.license_plate_pub.publish(str('TeamRed,multi12,-1,XR58'))
    #     #   self.end = True
    #     #   print("ended timer")
    #     pred = np.argmax(pred_arr)
          
    #     if self.num_of_crosswalks > 1:
    #       if (pred == 0):
    #         self.twist.linear.x = 0.09
    #         self.twist.angular.z = 1.0
    #       elif (pred == 1):
    #         self.twist.linear.x = 0.4
    #         self.twist.angular.z = 0.0
    #       elif (pred == 2):
    #         self.twist.linear.x = 0.09
    #         self.twist.angular.z = -1.0

    #       self.cmd_vel_pub.publish(self.twist)
    #     else:
    #       next_prob = pred_arr[0][1]
    #       p_scaled = min(-0.125/(next_prob - 1.0001), 1.0)
    #       prev_speed = self.twist.linear.x
    #       if (pred == 0):
    #         self.twist.linear.x = 0.09
    #         self.twist.angular.z = 1.3 #1.3 good before
    #       elif (pred == 1):
    #         self.twist.linear.x = min(prev_speed + 0.1, p_scaled) #0.6 normally
    #         self.twist.angular.z = 0.0
    #       elif (pred == 2):
    #         self.twist.linear.x = 0.09
    #         self.twist.angular.z = -1.3

    #       self.cmd_vel_pub.publish(self.twist)          

        

  def initial_turn(self):
    
    time.sleep(1)    
    self.twist.linear.x = 0.5
    self.twist.angular.z = 0.0
    self.cmd_vel_pub.publish(self.twist)
    time.sleep(0.75)
    self.twist.linear.x = 0.0
    self.twist.angular.z = 1.0
    self.cmd_vel_pub.publish(self.twist)
    time.sleep(2.1)
    self.twist.linear.x = 0.0
    self.twist.angular.z = 0.0
    self.cmd_vel_pub.publish(self.twist)
    time.sleep(2)
    self.finished_manuever = True
    self.start = False
    
def on_press(key):
    global end_global
    try:
        if key.char == 'n':
          print('here')
          end_global = True
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False

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