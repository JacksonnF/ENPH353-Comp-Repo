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
    self.been_on_sand = 0
    self.loop_count = 0
    self.enter_inner_loop = False
    self.align_cross_inner = False
    self.inner_manuever = False
    self.fucked = False
    self.get_to_sand = False
    


    # Stuff for pedestrian routine
    self.lastCentroid = (None,None)
    self.lastPedestrianSpeed = None

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
    
    next_line = None
    if lines is not None:
        for line in lines:
          x1, y1, x2, y2 = line[0]
          slope = (y2-y1)/(x2-x1)
          if slope < 5:
            next_line = line
        # x1, y1, x2, y2 = lines[0][0]
        x1, y1, x2, y2 = next_line[0]
        deg = np.rad2deg(np.arctan((y2-y1)/(x2-x1)))
        # print(deg)
        if deg > 1:
            self.twist.linear.x = 0.0
            self.twist.angular.z = -0.6
        elif deg < -1:
            self.twist.linear.x = 0.0
            self.twist.angular.z = 0.6
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
          if self.align_cross_inner:
            self.align_cross_inner = False
            self.inner_manuever = True
          #  self.predict_sand(bottom_half_rgb)
        
    if lines is None:
      print('no lines found')
      self.waiting += 1
      if self.waiting > 3:
        if self.align_cross_inner:
          self.align_cross_inner = False
          self.inner_manuever = True
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
    mask = cv2.inRange(filtered, np.array([0, 52, 128]), np.array([0, 255, 255]))
    eroded = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)
    dilated = cv2.dilate(eroded, np.ones((3, 3), np.uint8), iterations=3)
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
    start = time.time() * 1000
    interpreter = tf.lite.Interpreter(model_path="/home/fizzer/ros_ws/src/controller_pkg/data/model_3000c_quantized.tflite")
    interpreter.allocate_tensors()
    # print(time.time() * 1000 - start)
    # print(start)
    # Set the input tensor.
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_data)

    # Run inference.
    interpreter.invoke()

  # Get the output tensor.
    output_index = interpreter.get_output_details()[0]["index"]
    output_data = interpreter.get_tensor(output_index)
    # print(time.time() * 1000 - start)
    # Print the predicted class.
    predicted_class = np.argmax(output_data)
    # print(predicted_class)
    return output_data
  
  def predict_sand(self, img):
    img_aug = np.expand_dims(img, axis=0)
    # input_data = np.expand_dims(np.array(img_aug, dtype=np.float32), axis=-1)
    input_data = np.array(img_aug, dtype=np.float32)
    interpreter = tf.lite.Interpreter(model_path="/home/fizzer/ros_ws/src/controller_pkg/data/model_400aa_quantized.tflite") #400aa best so far
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
    
  def check_for_sand_end(self, img):
    w = img.shape[1]
    h = img.shape[0]
    cropped_img = img[h-240:h, 0:w]
    blur = cv2.GaussianBlur(cropped_img, (5, 5), 0)
    filtered = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(filtered, np.array([0, 0, 75]), np.array([5, 5, 90]))
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
      area1 = cv2.contourArea(contours[0])
      return area1
    else:
      return 0
  def check_for_sand_start(self, img):
    w = img.shape[1]
    h = img.shape[0]
    cropped_img = img[h-240:h, 0:w]
    blur = cv2.GaussianBlur(cropped_img, (5, 5), 0)
    filtered = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(filtered, np.array([13, 72, 91]), np.array([37, 166, 183]))
    area = cv2.countNonZero(mask)
    return area
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
    if self.start:
      self.spawn_position(-0.85, 0 , 0.5, 0,0,1,0)
      self.initial_turn()
      print('here')
      self.license_plate_pub.publish(str('TeamRed,multi12,0,XR58'))
      self.start = False
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
      self.prev_img = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    if self.fucked:
      print("fucked, turning left")
      self.twist.angular.z = 1.0
      self.twist.linear.x = 0.0
      self.cmd_vel_pub.publish(self.twist)
      bin_img = self.crop_for_prediction(cv_image, False)
      if np.mean(bin_img) != 0.0:
        self.fucked = False
      return

    if self.enter_inner_loop:
      if self.align_cross_inner:
        self.align_robot(cv_image)
        return
      if self.inner_manuever:
        self.inner_manuever_func()
        self.license_plate_pub.publish(str('TeamRed,multi12,-1,XR58'))
        return
      return
    
    if self.stop_at_crosswalk:
      self.align_robot(cv_image)
      return
       
    if self.waiting_to_cross:
      # print('cross')
      #  gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
      #  print(np.mean(self.prev_img - gray)**2)
      #  self.prev_img = gray
      hsvPedestrian = self.hsv_pedestrian(cv_image)
      contours = self.contour_pedestrian(hsvPedestrian)
      centroid = self.find_contour_centroid(contours[0])

      #Last centroid initialized but lastSpeed is not
      if(self.lastPedestrianSpeed==None and self.lastCentroid!=(None,None)):
        self.lastPedestrianSpeed = centroid[0]-self.lastCentroid[0]
        return

      #Last Centroid not initialized but lastSpeed is
      if(self.lastCentroid==(None,None)):
        self.lastCentroid = centroid
        return
      else:
        #Last centroid and last speed initialized
        currentPedestrianSpeed = centroid[0] - self.lastCentroid[0]

      # print("current speed: ", currentPedestrianSpeed)
      # print("last speed: ", self.lastPedestrianSpeed)

      if(620<centroid[0] and centroid[0]<660):
          self.waiting_to_cross = False
          self.crossing = True
          self.lastCentroid = (None,None)
          self.lastPedestrianSpeed = None
      else:
        self.lastCentroid = centroid      
        self.lastPedestrianSpeed = currentPedestrianSpeed   

      return
    
    #drive forward until both crosswalks are out of view
    if self.crossing:
      # print('crossing')
      contours = self.check_crosswalk_dist(cv_image)
      # print(contours)
      if contours[0] < 150 and contours[1] < 150:
        self.crossing = False
      else:
        # ####
        # self.twist.linear.x = 0.5
        # self.twist.angular.z = 0.0
        # self.cmd_vel_pub.publish(self.twist)
        
        # ####
        if self.num_of_crosswalks == 1:
          bin_img = self.crop_for_prediction(cv_image, False)
          pred_arr = self.predict(bin_img)
          pred = np.argmax(pred_arr)
          next_prob = pred_arr[0][1]
          p_scaled = min(-0.125/(next_prob - 1.0001), 1.0) #originallly 0.125
          prev_speed = self.twist.linear.x
          if (pred == 0):
            self.twist.linear.x = 0.1
            self.twist.angular.z = 1.2 #1.3 good before
          elif (pred == 1):
            self.twist.linear.x = min(prev_speed + 0.1, p_scaled) #0.6 normally
            self.twist.angular.z = 0.0
          elif (pred == 2):
            self.twist.linear.x = 0.1
            self.twist.angular.z = -1.2
          self.cmd_vel_pub.publish(self.twist)
        elif self.num_of_crosswalks == 2:
          print('crossing')
          self.crossing = False
          self.get_to_sand = True
          self.twist.linear.x = 0.5
          self.twist.angular.z = 0.0
          self.cmd_vel_pub.publish(self.twist)
          return

      return
    
    if self.get_to_sand:
      self.twist.linear.x = 0.5
      self.twist.angular.z = 0.0
      self.cmd_vel_pub.publish(self.twist)
      area = self.check_for_sand_start(cv_image)
      if area > 200000:
        self.get_to_sand = False
      return

    
    areas = self.check_crosswalk_dist(cv_image)
    if len(areas) > 1:
      # print(areas)

      if areas[0] > 5000 and areas[1]>75 and time.time()-self.start_time>5:
        # self.align_robot(cv_image)
        
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        self.stop_at_crosswalk = True
          
        self.area_seen_twice = 0
        if self.loop_count > 0:
          self.enter_inner_loop = True
          self.align_cross_inner = True
          return
        else:
          self.stop_at_crosswalk = True
          self.area_seen_twice = 0
        # self.waiting_to_cross = True
        return

    #Check if sand has ended
    if self.num_of_crosswalks > 1:
      road_area = self.check_for_sand_end(cv_image)
      if road_area > 80000 and self.been_on_sand > 25:
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)
        print('sand ended')
        self.num_of_crosswalks = 0
        self.been_on_sand = 0
        self.loop_count += 1
        return

    
    if self.num_of_crosswalks > 1:
      bottom_half_rgb = self.crop_for_prediction(cv_image, True)
      pred_arr = self.predict_sand(bottom_half_rgb)
      self.been_on_sand += 1
    else:
      bin_img = self.crop_for_prediction(cv_image, False)
      if np.mean(bin_img) == 0.0:
        self.fucked = True
        return
      pred_arr = self.predict(bin_img)
    pred = np.argmax(pred_arr)
    if self.num_of_crosswalks > 1:
      # if (pred == 0):
      #   self.twist.linear.x = 0.1
      #   self.twist.angular.z = 1.2
      # elif (pred == 1):
      #   self.twist.linear.x = 0.5
      #   self.twist.angular.z = 0.0
      # elif (pred == 2):
      #   self.twist.linear.x = 0.1
      #   self.twist.angular.z = -1.2
      next_prob = pred_arr[0][1]
      p_scaled = min(-0.125/(next_prob - 1.0001), 0.8) #originallly 0.125
      prev_speed = self.twist.linear.x
      if (pred == 0):
        self.twist.linear.x = 0.1
        self.twist.angular.z = 1.3 #1.3 good before
      elif (pred == 1):
        self.twist.linear.x = min(prev_speed + 0.1, p_scaled) #0.6 normally
        self.twist.angular.z = 0.0
        # self.twist.linear.x = 1.0
      elif (pred == 2):
        self.twist.linear.x = 0.1
        self.twist.angular.z = -1.3





      self.cmd_vel_pub.publish(self.twist)
    else:
      
      prev_speed = self.twist.linear.x
      straightProb = pred_arr[0][1]
      leftProb =  pred_arr[0][0]
      rightProb = pred_arr[0][2]

      self.twist.linear.x = (prev_speed + np.power(straightProb,0.3)*1.75)/3
      # self.twist.linear.x = min(prev_speed + 0.1, np.power(straightProb,0.3)*1.5) #0.6 normally 
      self.twist.angular.z = np.sign(leftProb-rightProb)*np.power(np.abs((leftProb- rightProb)),0.8)*3.0
      # next_prob = pred_arr[0][1]
      # p_scaled = min(-0.125/(next_prob - 1.0001), 0.8) #originallly 0.125
      # prev_speed = self.twist.linear.x
      # if (pred == 0):
      #   self.twist.linear.x = 0.1
      #   self.twist.angular.z = 1.3 #1.3 good before
      # elif (pred == 1):
      #   self.twist.linear.x = min(prev_speed + 0.1, p_scaled) #0.6 normally
      #   self.twist.angular.z = 0.0
      #   # self.twist.linear.x = 1.0
      # elif (pred == 2):
      #   self.twist.linear.x = 0.1
      #   self.twist.angular.z = -1.3

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
    time.sleep(1.0)
    self.twist.linear.x = 0.5
    self.twist.angular.z = 0.0
    self.cmd_vel_pub.publish(self.twist)
    time.sleep(0.75)
    self.twist.linear.x = 0.0
    self.twist.angular.z = 1.5
    self.cmd_vel_pub.publish(self.twist)
    time.sleep(1.5)
    self.twist.linear.x = 0.0
    self.twist.angular.z = 0.0
    self.cmd_vel_pub.publish(self.twist)
    # time.sleep(2)
    self.finished_manuever = True
    self.start = False

  def inner_manuever_func(self):
    # time.sleep(1)    
    self.twist.linear.x = -0.5
    self.twist.angular.z = 0.0
    self.cmd_vel_pub.publish(self.twist)
    time.sleep(0.49)
    self.twist.linear.x = 0.0
    self.twist.angular.z = 1.0
    self.cmd_vel_pub.publish(self.twist)
    time.sleep(2.1)
    self.twist.linear.x = 0.35
    self.twist.angular.z = 0.0
    self.cmd_vel_pub.publish(self.twist)
    time.sleep(0.7)
    self.twist.linear.x = 0.0
    self.twist.angular.z = 0.0
    self.cmd_vel_pub.publish(self.twist)

    self.inner_manuever = False
    
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