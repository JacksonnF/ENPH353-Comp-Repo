#! /usr/bin/env python3
import rospy 
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2
from sensor_msgs.msg import Image
from rosgraph_msgs.msg import Clock
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

import numpy as np 
from pynput import keyboard
import tensorflow as tf
from tensorflow import keras
import time


depressed = 0
recording = False
timer_started = 0
spawn_at_start = False

class data_collector:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber('/R1/pi_camera/image_raw',
    Image,self.callback)
    self.cmd_vel_sub = rospy.Subscriber('/R1/cmd_vel', Twist, self.vel_callback)
    self.current_vel = Twist()
    self.twist = Twist()
    self.cmd_vel_pub = rospy.Publisher('/R1/cmd_vel', Twist, queue_size=1)

    self.license_plate = rospy.Publisher('/license_plate', String, queue_size=1)

    self.drive_imgs_gray = []
    self.drive_imgs_rgb = []
    self.vel_cmds = []
    self.flag = 30
    # self.model = keras.models.load_model('/home/fizzer/ros_ws/src/controller_pkg/data/model_20000a.h5') #model 20 is the best one
    # self.interpreter = tf.lite.Interpreter(model_path="/home/fizzer/ros_ws/src/controller_pkg/data/model_2000c_quantized.tflite")
    listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
    listener.start()

  def vel_callback(self, data):
    self.current_vel = data
  #og position -0.85, 0 , 0.5, 0,0,1,0
  #1.2x, .7y .5z orientation 0,0,1,1
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
    
  def predict(self, img):
    img_aug = np.expand_dims(img, axis=0)
    input_data = np.expand_dims(np.array(img_aug, dtype=np.float32), axis=-1)
    interpreter = tf.lite.Interpreter(model_path="/home/fizzer/ros_ws/src/controller_pkg/data/model_2000c_quantized.tflite")
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
    print(predicted_class)
    return predicted_class
  
  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    
    global spawn_at_start
    if spawn_at_start:
      self.spawn_position(-0.85, 0 , 0.5, 0,0,1,0)
      self.initial_turn()
      spawn_at_start = False
    
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    scale_percent = 20

    w = int(gray.shape[1] * scale_percent / 100)
    h = int(gray.shape[0] * scale_percent / 100)
    dim = (w, h)

    resized_img_rgb = cv2.resize(cv_image, dim, interpolation=cv2.INTER_AREA)
    resized_img_gray = cv2.resize(gray, dim, interpolation=cv2.INTER_AREA)

    #get bottom half of image
    bottom_half_gray = resized_img_gray[int(h/2):h, 0:w]
    bottom_half_rgb = resized_img_rgb[int(h/2):h, 0:w]

    ret, bin_img = cv2.threshold(bottom_half_gray, 180,255,0)
    # pred = self.model.predict(np.expand_dims(bin_img, axis=0))[0]
    # prediction = np.argmax(pred)
    # prediction = self.predict(bin_img)
    next_v = [self.current_vel.linear.x, self.current_vel.angular.z]
    real_val = np.argmax(next_v)
    if real_val == 0:
      real_val = 1
    else:
       if next_v[1] > 0:
         real_val = 0
       elif next_v[1] < 0:
         real_val = 2

    global recording
    if (recording == True):
      # print("prediction: ", prediction, "real value: ", real_val)

      self.drive_imgs_gray.append(bin_img)
      self.drive_imgs_rgb.append(bottom_half_rgb)
      self.vel_cmds.append([self.current_vel.linear.x, self.current_vel.angular.z])
    global depressed 
    if (depressed == 1):
       print('saving')
       depressed = 0
       np.save('/home/fizzer/ros_ws/src/controller_pkg/data/driving_data_1/data_vels'+str(self.flag)+
               '.npy', self.vel_cmds)
       np.savez_compressed('/home/fizzer/ros_ws/src/controller_pkg/data/driving_data_1/img_gray_comp'
                           +str(self.flag)+'.npz', *self.drive_imgs_gray)
       np.savez_compressed('/home/fizzer/ros_ws/src/controller_pkg/data/driving_data_1/img_rgb_comp'
                           +str(self.flag)+'.npz', *self.drive_imgs_rgb)
       print('saved')
       recording = False
       self.drive_imgs_rgb = []
       self.drive_imgs_gray = []
       self.vel_cmds = []
       self.flag += 1

    # cv2.imshow("bin window", bin_img)
    cv2.imshow("rgb window", bottom_half_rgb)

    cv2.waitKey(1)

def on_press(key):
    try:
        global depressed
        global recording
        global timer_started
        global spawn_at_start
        if key.char == 't' and timer_started == 0:
          timer_started = 1
        elif key.char == 't' and timer_started == 2:
          timer_started = 3
        if key.char == 'b':
          depressed = 1
        if key.char == 'p': #start recording data
           recording = True
           print('recording')
        if key.char == 'n':
           spawn_at_start = True 
           
           
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    if key == keyboard.Key.esc:
        # Stop listener
        return False
  

def main():
    ic = data_collector()
    rospy.init_node('data_collector', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()