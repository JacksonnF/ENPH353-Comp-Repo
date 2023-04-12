#!/usr/bin/env python3
import rospy
from gazebo_msgs.srv import SetModelState, GetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Quaternion
import numpy as np
import time
# from tensorflow import keras
import tensorflow as tf
from ordered_set import OrderedSet
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import subprocess
import roslaunch
import csv
from pynput import keyboard
import collections
from std_msgs.msg import String


collected_plates_arr = []


def hsvImage(img):
    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the color
    color1HSV = np.array([0,0,102])
    color2HSV = np.array([0,0,122])
    color3HSV = np.array([0,0,201])
    colorVariance = np.array([2,2,2])

    # Apply the HSV filter
    mask = cv2.inRange(hsv, color1HSV-colorVariance, color1HSV+colorVariance)
    mask2 = cv2.inRange(hsv, color2HSV-colorVariance, color2HSV+colorVariance)
    mask3 = cv2.inRange(hsv, color3HSV-colorVariance, color3HSV+colorVariance)

    # Erode the masks to remove noise
    erodeIterations = 3
    kernel1 = np.ones((2,2), np.uint8)
    eroded1 = cv2.erode(mask, kernel1, iterations=erodeIterations)
    eroded2 = cv2.erode(mask2, kernel1, iterations=erodeIterations)
    eroded3 = cv2.erode(mask3, kernel1, iterations=erodeIterations)
    
    # Combine the HSV masks into a single mask
    combinedResult1 = cv2.bitwise_or(eroded1, eroded2)
    combinedResult2 = cv2.bitwise_or(combinedResult1, eroded3)

    # Dilate Horizontally
    kernel3 = np.ones((1,2), np.uint8)
    dilatedHorizontally  = cv2.dilate(combinedResult2,kernel3, iterations=1)

    erodedHorizontally  = cv2.dilate(dilatedHorizontally,kernel3, iterations=1)
    
    # Dilate Vertically
    kernel2 = np.ones((5,1), np.uint8)
    dilatedVertically  = cv2.dilate(dilatedHorizontally,kernel2, iterations=15)

    #Erode again
    finalEroded = cv2.erode(dilatedVertically,kernel2, iterations=15)

    blurred = cv2.GaussianBlur(finalEroded, (5,5), 0)

    return blurred

def contours(thresh_img):
    height, width = 720, 1280
    binary_image = np.zeros((height,width), dtype=np.uint8)

    sucessful = True

    # Find the contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Aproximate the contours
    approx_contours = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
        approx_contours.append(approx)

    filtered_contours = [c for c in approx_contours if cv2.contourArea(c) > 8000]

    if(len(filtered_contours)>0):
        # Finds the largest and second largest contours by area
        currentMaxArea = 0
        index = 0
        for i in range(0,len(filtered_contours)):
            currentArea = cv2.contourArea(filtered_contours[i])
            if(currentArea>currentMaxArea):
                index = i
                currentMaxArea = currentArea

        # Puts the largest contour into a list
        filtered_contours2 = [filtered_contours[index]]

        # Draws the contours onto the binary image
        cv2.drawContours(binary_image, filtered_contours2, -1, (255), -1)

        return binary_image, filtered_contours2, sucessful
    else:
        return None, None, False

def find_outermost_points_combined(contours, n=1):
    # Combine all contours into a single list
    combined_contour = np.vstack(contours)

    # Calculate the centroid of the combined contour
    M = cv2.moments(combined_contour)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    
    # Initialize the lists for outermost points
    topLeft, bottomLeft, topRight, bottomRight = [], [], [], []
    
    # Find the outermost points
    for point in combined_contour:
        x, y = point[0]
        
        # Add the point to the corresponding list
        topLeft.append((x, y, x + y))
        bottomLeft.append((x, y, x - y))
        topRight.append((x, y, y - x))
        bottomRight.append((x, y, x + y))
    
    # Sort the lists and get the n most outermost points
    topLeft = sorted(topLeft, key=lambda p: p[2])[:n]
    bottomLeft = sorted(bottomLeft, key=lambda p: p[2], reverse=True)[:n]
    topRight = sorted(topRight, key=lambda p: p[2], reverse=True)[:n]
    bottomRight = sorted(bottomRight, key=lambda p: p[2], reverse=True)[:n]
    
    # Calculate the average position for each point
    avg_topLeft = (sum(p[0] for p in topLeft) / n, sum(p[1] for p in topLeft) / n)
    avg_bottomLeft = (sum(p[0] for p in bottomLeft) / n, sum(p[1] for p in bottomLeft) / n)
    avg_topRight = (sum(p[0] for p in topRight) / n, sum(p[1] for p in topRight) / n)
    avg_bottomRight = (sum(p[0] for p in bottomRight) / n, sum(p[1] for p in bottomRight) / n)

    return tuple(map(int, avg_topLeft)), tuple(map(int, avg_bottomLeft)), tuple(map(int, avg_topRight)), tuple(map(int, avg_bottomRight))

def crop_and_transform(img, topLeft, bottomLeft, topRight, bottomRight, height=500, width=400):
    transformPointsBefore = np.array([topLeft,topRight,bottomRight,bottomLeft])
    transformPointsBefore = np.array(transformPointsBefore, dtype=np.float32)
    output_points = np.array([(0, 0), (width - 1, 0), (width - 1, height - 1),(0, height - 1), ], dtype=np.float32)
    # Compute the perspective transform matrix (homography)
    M = cv2.getPerspectiveTransform(transformPointsBefore, output_points)
    # Apply the perspective transformation to crop the region of interest
    cropped_img = cv2.warpPerspective(img, M, (width, height))
    return cropped_img


# LETTER ISOLATION STARTS HERE

def hsvLetters(img):
    # Convert BGR to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the color
    color1HSV = np.array([113.37,229.5,99.45])
    colorVariance = np.array([140,140,140])

    # Apply the HSV filter
    mask = cv2.inRange(hsv, color1HSV-colorVariance, color1HSV+colorVariance)

    # Erode the masks to remove noise
    erodeIterations = 3
    kernel1 = np.ones((2,2), np.uint8)
    eroded1 = cv2.erode(mask, kernel1, iterations=erodeIterations)
    
    blurred = cv2.GaussianBlur(eroded1, (3,3), 0)

    return blurred

def letter_contours(thresh_img):
    height, width = 500, 400
    binary_image = np.zeros((height,width), dtype=np.uint8)

    sucessful = True

    # Find the contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # # Aproximate the contours
    # approx_contours = []
    # for cnt in contours:
    #     approx = cv2.approxPolyDP(cnt, 0.001 * cv2.arcLength(cnt, True), True)
    #     approx_contours.append(approx)

    filtered_contours = [c for c in contours if cv2.contourArea(c) > 200]

    if len(filtered_contours) < 1:
        sucessful = False
        return None, None, sucessful

    # Draws the contours onto the binary image
    cv2.drawContours(binary_image, filtered_contours, -1, (255), -1)

    return binary_image, filtered_contours, sucessful

def find_letter_corners(contours):
    # Concatenate all contours into a single array
    if(len(contours)>1):
        all_points = np.vstack(contours)
    elif(len(contours)==1):
        all_points = contours[0]
    else: return None, None, None, None
    
    # Find the minimum area rectangle for the concatenated points
    min_area_rect = cv2.minAreaRect(all_points)
    
    # Convert min_area_rect to a box with 4 vertices
    box = cv2.boxPoints(min_area_rect)
    box = np.int0(box)

    # Sort corners by their y-coordinate
    sorted_corners = sorted(box, key=lambda corner: corner[1])
    
    # Separate the corners into top and bottom pairs
    top_corners = sorted(sorted_corners[:2], key=lambda corner: corner[0])
    bottom_corners = sorted(sorted_corners[2:], key=lambda corner: corner[0], reverse=True)
    
    # Combine the corners in the desired order
    topLeft, bottomLeft = top_corners
    topRight, bottomRight = bottom_corners

    if(1<topLeft[0]<39 and 340<topRight[0]<370):
        return (tuple(topLeft), tuple(bottomLeft), tuple(topRight), tuple(bottomRight))
    else:
        return None, None, None, None

def find_letter_corners2(contours):
    # Concatenate all contours into a single array
    all_points = np.vstack(contours)
    
    # Find the minimum area rectangle for the concatenated points
    x,y,w,h = cv2.boundingRect(all_points)

    top_left = (x, y)
    bottom_left = (x, y + h)
    top_right = (x + w, y)
    bottom_right = (x + w, y + h)

    return np.array([top_left, bottom_left, top_right, bottom_right], dtype="int")
   

def find_individual_letter_corners(contours):

    rectangles = []
    # if(len(contours)==4):
    #     for i in range(0,len(contours)):
    #         rectangles.append(find_letter_corners2([contours[i]]))
    #     # Sort rectangles from left to right
    #     rectangles.sort(key=lambda x: x[:, 0].min())
    #     return True, rectangles
    # else:
    #     return False, None

    for i in range(0,len(contours)):
        rectangles.append(find_letter_corners2([contours[i]]))
        # Sort rectangles from left to right
        rectangles.sort(key=lambda x: x[:, 0].min())
    return True, rectangles


    
def splitRectangles(rectangles):
    # Loop through the rectangles list using index
    i = 0
    while i < len(rectangles):
        # Calculate the width of the rectangle
        rect = rectangles[i]
        width = abs(rect[0][0] - rect[2][0])

        # If the width is greater than 30 pixels, remove the rectangle and split it into two
        if width > 60:
            rectangles.pop(i)
            half_width = width / 2

            # Create two new rectangles with half the width
            rect1 = np.array([[rect[0][0], rect[0][1]], [rect[0][0], rect[1][1]], [rect[0][0] + half_width, rect[0][1]], [rect[0][0] + half_width, rect[1][1]]], dtype="int")
            rect2 = np.array([[rect[0][0] + half_width, rect[0][1]], [rect[0][0] + half_width, rect[1][1]], [rect[2][0], rect[2][1]], [rect[3][0], rect[3][1]]], dtype="int")

            # Insert the new rectangles in the original list position
            rectangles.insert(i, rect1)
            rectangles.insert(i + 1, rect2)
        else:
            i += 1
    return rectangles

def carNumberHsv(img):
    # Convert the image to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Define the lower and upper bounds of the color
    lower = np.array([0, 0, 0])
    upper = np.array([0, 0, 85])
    
    # Create a mask for the color
    mask = cv2.inRange(hsv, lower, upper)

    eroded = cv2.erode(mask, np.array([2,2]), iterations=2)
    
    return eroded

def find_letter_corners_carNumber(contours):
    # Concatenate all contours into a single array
    if(len(contours)>1):
        all_points = np.vstack(contours)
    elif(len(contours)==1):
        all_points = contours[0]
    else: return None, None, None, None
    
    # Find the minimum area rectangle for the concatenated points
    min_area_rect = cv2.minAreaRect(all_points)
    
    # Convert min_area_rect to a box with 4 vertices
    box = cv2.boxPoints(min_area_rect)
    box = np.int0(box)

    # Sort corners by their y-coordinate
    sorted_corners = sorted(box, key=lambda corner: corner[1])
    
    # Separate the corners into top and bottom pairs
    top_corners = sorted(sorted_corners[:2], key=lambda corner: corner[0])
    bottom_corners = sorted(sorted_corners[2:], key=lambda corner: corner[0], reverse=True)
    
    # Combine the corners in the desired order
    topLeft, bottomLeft = top_corners
    topRight, bottomRight = bottom_corners

    # if(1<topLeft[0]<39 and 340<topRight[0]<370):
    #     return (tuple(topLeft), tuple(bottomLeft), tuple(topRight), tuple(bottomRight))
    # else:
    #     return None, None, None, None

    return (tuple(topLeft), tuple(bottomLeft), tuple(topRight), tuple(bottomRight))



bridge = CvBridge()
current_frame = None

def isolatePlate(img):
    
    # Find pack of car
    blurredImg = cv2.GaussianBlur(img, (5, 5), 64)
    beforeImage = hsvImage(blurredImg)
    plateHsv, filtered_contours, sucessful = contours(beforeImage)
    
    if(sucessful==False):
        # print("sucessful: ", sucessful)
        return None,None, False
    
    topLeft, topRight, bottomLeft, bottomRight = find_outermost_points_combined(filtered_contours)

    inwardShift = 1
    topLeft = (topLeft[0] + inwardShift, topLeft[1] + inwardShift)
    topRight = (topRight[0] - inwardShift, topRight[1] + inwardShift)
    bottomLeft = (bottomLeft[0] + inwardShift, bottomLeft[1] - inwardShift)
    bottomRight = (bottomRight[0] - inwardShift, bottomRight[1] - inwardShift)
    croppedImage = crop_and_transform(img, topLeft, bottomLeft, topRight, bottomRight)
    # cv2.imshow("PlateCrop", croppedImage)
    # cv2.waitKey(1)
    # ------------------------------------------------------------

    #Isolate Car Number --------------------------------------

    isolatedCarNumber = carNumberHsv(croppedImage)
    # cv2.imshow("CarNumber", isolatedCarNumber)
    # cv2.waitKey(1)

    plateHsv3, filtered_contours_letters3, sucessful_letters3 = letter_contours(isolatedCarNumber)

    if(sucessful_letters3==False):
        # print("sucessful_letters: n ", sucessful_letters)
        return None, None, False
    topLeft3, topRight3, bottomRight3, bottomLeft3 = find_letter_corners_carNumber(filtered_contours_letters3)

    if(topLeft3==None):
        # print("None returned")
        return None, None, False
    
    outwardShift3 = 1
    topLeft3 = (topLeft3[0] - outwardShift3, topLeft3[1] - outwardShift3)
    topRight3 = (topRight3[0] + outwardShift3, topRight3[1] - outwardShift3)
    bottomLeft3 = (bottomLeft3[0] - outwardShift3, bottomLeft3[1] + outwardShift3)
    bottomRight3 = (bottomRight3[0] + outwardShift3, bottomRight3[1] + outwardShift3)

    croppedLetters3 = crop_and_transform(isolatedCarNumber, topLeft3, bottomLeft3, topRight3, bottomRight3, height = 40, width = 200)

    # # Draw circles at topLeft3, topRight3, bottomLeft3, and bottomRight3 on img
    # cv2.circle(croppedImage, topLeft3, 5, (0, 0, 255), -1)
    # cv2.circle(croppedImage, topRight3, 5, (0, 0, 255), -1)
    # cv2.circle(croppedImage, bottomLeft3, 5, (0, 0, 255), -1)
    # cv2.circle(croppedImage, bottomRight3, 5, (0, 0, 255), -1)

    # cv2.imshow("circle", croppedImage)
    # cv2.waitKey(1)


    # Isolate Plate Letters -----------------------------------------
    isolatedLetters = hsvLetters(croppedImage)
    plateHsv2, filtered_contours_letters, sucessful_letters = letter_contours(isolatedLetters)
    # print("sucessful_letters: ", sucessful_letters)
    if(sucessful_letters==False):
        # print("sucessful_letters: ", sucessful_letters)
        return None, None, False
    topLeft2, topRight2, bottomRight2, bottomLeft2 = find_letter_corners(filtered_contours_letters)

    # print("topLeft2: ", topLeft2)

    if(topLeft2==None):
        # print("None returned")
        return None, None, False
    
    outwardShift = 1
    topLeft2 = (topLeft2[0] - outwardShift, topLeft2[1] - outwardShift)
    topRight2 = (topRight2[0] + outwardShift, topRight2[1] - outwardShift)
    bottomLeft2 = (bottomLeft2[0] - outwardShift, bottomLeft2[1] + outwardShift)
    bottomRight2 = (bottomRight2[0] + outwardShift, bottomRight2[1] + outwardShift)

    croppedLetters = crop_and_transform(isolatedLetters, topLeft2, bottomLeft2, topRight2, bottomRight2, height=40, width=200)


    return croppedLetters, croppedLetters3, True


def isolatePlate2(croppedLetters):

    # ------------------------------------------------------------

    # Crop individual letters

    plateHsv3, filtered_contours_letters_individual , sucessful_letters3 = letter_contours(croppedLetters)
    # binaryImage0 = np.zeros((croppedLetters.shape[0], croppedLetters.shape[1], 3), np.uint8)
    # binaryImage = np.zeros((croppedLetters.shape[0], croppedLetters.shape[1], 3), np.uint8)
    # binaryImage2 = np.zeros((croppedLetters.shape[0], croppedLetters.shape[1], 3), np.uint8)
    
    success4, rectangles = find_individual_letter_corners(filtered_contours_letters_individual)

    # print("rectangles: ", len(rectangles))

    adjustedRectangles = splitRectangles(rectangles)
    if(len(adjustedRectangles)!=4):
        return None
    # print("adjustedRectangles: ", len(adjustedRectangles))
    
    if(success4 == True):
        croppedLetters_individual = [None, None, None, None]
        for i in range(0,len(rectangles)):
            croppedLetters_individual[i] = crop_and_transform(croppedLetters, adjustedRectangles[i][0], adjustedRectangles[i][1], adjustedRectangles[i][2], adjustedRectangles[i][3], height=40, width=32)
            if(type(croppedLetters_individual)==type(None)):
                return None
    else:
        # print("success4: ",  success4)
        croppedLetters_individual = None

    return croppedLetters_individual

def isolatePlate2_carNumber(croppedLetters):

    # ------------------------------------------------------------

    # Crop individual letters

    plateHsv3, filtered_contours_letters_individual , sucessful_letters3 = letter_contours(croppedLetters)
    # binaryImage0 = np.zeros((croppedLetters.shape[0], croppedLetters.shape[1], 3), np.uint8)
    # binaryImage = np.zeros((croppedLetters.shape[0], croppedLetters.shape[1], 3), np.uint8)
    # binaryImage2 = np.zeros((croppedLetters.shape[0], croppedLetters.shape[1], 3), np.uint8)
    
    success4, rectangles = find_individual_letter_corners(filtered_contours_letters_individual)

    # print("rectangles: ", len(rectangles))

    # adjustedRectangles = splitRectangles(rectangles)
    if(len(rectangles)!=2):
        return None
    # print("adjustedRectangles: ", len(adjustedRectangles))

    adjustedRectangles = rectangles
    
    if(success4 == True):
        croppedNumber = crop_and_transform(croppedLetters, adjustedRectangles[1][0], adjustedRectangles[1][1], adjustedRectangles[1][2], adjustedRectangles[1][3], height=40, width=32)
        # croppedLetters_individual = [None, None, None, None]
        # for i in range(0,len(rectangles)):
        #     croppedLetters_individual[i] = crop_and_transform(croppedLetters, adjustedRectangles[i][0], adjustedRectangles[i][1], adjustedRectangles[i][2], adjustedRectangles[i][3], height=40, width=32)
        #     if(type(croppedLetters_individual)==type(None)):
        #         return None
    else:
        # print("success4: ",  success4)
        croppedNumber = None

    return croppedNumber

def predictLetter(croppedLetter):

    # print(boolean)

    # predictedArray = letModel.predict(np.expand_dims(croppedLetter, axis=0))
    # predictedLetter = letters[np.argmax(predictedArray)]
    img_aug = np.expand_dims(croppedLetter, axis=0)
    input_data = np.expand_dims(np.array(img_aug, dtype=np.float32), axis=-1)

    interpreter = tf.lite.Interpreter('/home/fizzer/ros_ws/src/controller_pkg/data/let_model_4_quantized.tflite')
    interpreter.allocate_tensors()

    # Set the input tensor.
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_data)

    # Run inference.
    interpreter.invoke()

    # Get the output tensor.
    output_index = interpreter.get_output_details()[0]["index"]
    output_data = interpreter.get_tensor(output_index)

    # # Print the predicted class.
    # predicted_class = np.argmax(output_data)
    return letters[np.argmax(output_data)]

def predictNumber(croppedNumber):
    # print(boolean)

    # predictedArray = numModel.predict(np.expand_dims(croppedNumber, axis=0))
    # predictedNumber = numbers[np.argmax(predictedArray)]
    img_aug = np.expand_dims(croppedNumber, axis=0)
    input_data = np.expand_dims(np.array(img_aug, dtype=np.float32), axis=-1)

    interpreter = tf.lite.Interpreter('/home/fizzer/ros_ws/src/controller_pkg/data/num_model_4.2_quantized.tflite')
    interpreter.allocate_tensors()

    # Set the input tensor.
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_data)

    # Run inference.
    interpreter.invoke()

    # Get the output tensor.
    output_index = interpreter.get_output_details()[0]["index"]
    output_data = interpreter.get_tensor(output_index)

    # # Print the predicted class.
    # predicted_class = np.argmax(output_data)
    return numbers[np.argmax(output_data)]
    

def predictCarNumber(croppedNumber):
    # print(boolean)

    # predictedArray = carNumModel.predict(np.expand_dims(croppedNumber, axis=0))
    # predictedNumber = carNumbers[np.argmax(predictedArray)]
    img_aug = np.expand_dims(croppedNumber, axis=0)
    input_data = np.expand_dims(np.array(img_aug, dtype=np.float32), axis=-1)

    interpreter = tf.lite.Interpreter('/home/fizzer/ros_ws/src/controller_pkg/data/car_num_model_1_quantized.tflite')
    interpreter.allocate_tensors()

    # Set the input tensor.
    input_index = interpreter.get_input_details()[0]["index"]
    interpreter.set_tensor(input_index, input_data)

    # Run inference.
    interpreter.invoke()

    # Get the output tensor.
    output_index = interpreter.get_output_details()[0]["index"]
    output_data = interpreter.get_tensor(output_index)

    # # Print the predicted class.
    # predicted_class = np.argmax(output_data)
    return carNumbers[np.argmax(output_data)]
    

def predictPlate(cropped_letters):

    licenseString = ""
    for i in range(0,2):
        letter =  predictLetter(cropped_letters[i])
        licenseString += letter

    for i in range(2,4):
        letter =  predictNumber(cropped_letters[i])
        licenseString += letter

    return licenseString



def callback(data):
    # print("Callback")
    global carNumber
    global current_frame
    global current_croppedLetters_individual
    global count
    global queue1
    global queue2
    global processedPlateStrings
    global plateReadings
    global skipIsolation
    global firstPlate8Time

    try:
        # Convert ROS Image message to OpenCV image
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {}".format(e))

    # Store current frame
    current_frame = cv_image


    if(skipIsolation == False):
        output1, output11, successful = isolatePlate(current_frame)
        # cv2.imshow("Output11", output11)
        # cv2.waitKey(1)
        if(successful==True):
            queue1.put((output1,output11))
            return
        
        if(not queue1.empty()):
            queue1Element = queue1.get()
            output2 = isolatePlate2(queue1Element[0])
            output22 = isolatePlate2_carNumber(queue1Element[1])
            if(type(output2)!=type(None) and type(output22)!=type(None)):

                # filename = "./data/"+str(carNumber)+"-{}-{}.jpg".format(count, session)
                # cv2.imwrite(filename, output22)
                # rospy.loginfo("Saved image {}".format(filename))
                # count = count+1

                queue2.put((output2,output22))
            return
        
        if(not queue2.empty()):
            queue2Element = queue2.get()
            output3 = predictPlate(queue2Element[0])
            output33 = predictCarNumber(queue2Element[1])
            if (int(output33) == 8 and type(firstPlate8Time)==type(None)):
                firstPlate8Time = time.time()
            plateReadings[(int(output33))-1].append(output3)

    if(len(plateReadings[7])>3 and queue1.empty() and queue2.empty() and skipIsolation == False and type(firstPlate8Time)!=type(None) and time.time()-firstPlate8Time>1):
        print(plateReadings)
        print("")
        global license_plate_pub    
        rate = rospy.Rate(8)
        for i in range(0,len(plateReadings)):
            if (len(plateReadings[i])!=0):
                count = collections.Counter(plateReadings[i])
                most_common = count.most_common(1)
                print("Plate " + str(i+1) + ": " + most_common[0][0] + ", numImages: " + str(count[most_common[0][0]]))
                print(str('TeamRed,multi12,'+str(count[most_common[0][0]])+','+most_common[0][0]))
                license_plate_pub.publish(str('TeamRed,multi12,'+str(i+1)+','+most_common[0][0]))
                rate.sleep()
        license_plate_pub.publish(str('TeamRed,multi12,-1,XR58')) 
        skipIsolation = True


def on_press(key):
    global count
    global session
    global carNumber
    global skipIsolation
    global plateReadings
    try:
        key_num = int(key.char)
    except (AttributeError, ValueError):
        return

    # if 7 <= key_num <= 8:
    #     print('You pressed {}'.format(key_num))
    #     print(licensePlateList)
    #     skipIsolation = True
        

    if 1 <= key_num <= 6:
        print('You pressed {}'.format(key_num))
        # Save the image with the specified file name
        carNumber += 1
        print("carNumber: ", carNumber)
           

if __name__ == '__main__':
    import queue
    global carNumber
    carNumber = 1
    global session
    global queue1
    global queue2
    global processedPlateStrings
    global skipIsolation
    skipIsolation = False
    global reportPlates
    reportPlates = False
    global firstPlate8Time
    
    global plateReadings
    global license_plate_pub
    plateReadings = [[],[],[],[],[],[], [], []]

    firstPlate8Time = None

    processedPlateStrings = []
    queue1 = queue.Queue()
    queue2 = queue.Queue()

    time.sleep(1)
    rospy.init_node('image_isolator')
    image_sub = rospy.Subscriber('/R1/pi_camera/image_raw', Image, callback)
    license_plate_pub = rospy.Publisher('/license_plate', String, 
                                             queue_size=1)
    licensePlateList = []

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    numbers = "0123456789"
    carNumbers = "12345678"
    # letModel = keras.models.load_model("/home/fizzer/ros_ws/src/controller_pkg/data/let_model_4.h5")
    # numModel = keras.models.load_model("/home/fizzer/ros_ws/src/controller_pkg/data/num_model_4.2.h5")
    # carNumModel = keras.models.load_model("/home/fizzer/ros_ws/src/controller_pkg/data/car_num_model_1.h5")
    # Open the CSV file
    with open('/home/fizzer/ros_ws/src/2022_competition/enph353/enph353_gazebo/scripts/plates.csv', newline='') as csvfile:
        # Create a CSV reader object
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        # Read each row in the CSV file
        for row in reader:
            licensePlateList.append(str(row))
    print(licensePlateList)
        

    # Setup the key listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    global current_croppedLetters_individual
    current_croppedLetters_individual = [None,None,None,None]

    global count
    count = 0
    
    rospy.spin()