#!/usr/bin/env python

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# numberRekognitionColor.py
# Santiago Perez Torres
# 01/05/20
# This file contains a class that edits an image and cut the backgroud of
# the number (the element that we are interested)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import os
import time

class numberRekognition():  

    def __init__(self):
        """
        This method starts the connection with the robot camera to get the image from it.
        """
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/turtlebot3/camera/image_raw", Image)
        #time.sleep(1)


    def process_image(self, data):
        """
        This method gets the information taken from the robot camera.

        Args:
            data (image):

        """

        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.loginfo(e)
        
        #Extract number from the image
        extract_number(img)
    
        return True

        
    def extract_number(self, img):
        """
        This method extracts the number from the robot camera and saves it in a folder.

        Args:
            img (image):

        """

        
        cv2.imshow("bw", img)
        cv2.waitKey(0)

        # Getting path
        path = os.getcwd() + '/catkin_ws/src/cookobot-rekognition/images'

        # Converting image from BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Defining the range of the color of the table number
        lower_color = np.array([0, 0, 0])
        upper_color = np.array([58, 51, 255])

        # Creating mask
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Size of the image
        height, width, channels = img.shape

        # Calculating blob centroide
        M = cv2.moments(mask, False)
        try:
            cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
        except ZeroDivisionError:
            cy, cx = height/2, width/2

        # Searching contours color objects
        _, contornos, __ = cv2.findContours(
            mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)

        for contorno in contornos:
            tam_contorno = cv2.contourArea(contorno)
            tam_mascara = mask.size
            x = (tam_contorno*100)/tam_mascara

            # Cropping table number
            x, y, w, h = cv2.boundingRect(contorno)
            roi = img[y:y + h, x:x + w]

            # Make black and white
            h, s, v = cv2.split(roi)

            # Make binary
            blur = cv2.GaussianBlur(v, (5, 5), 0)
            ret3, blackAndWhiteImage = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # Save image
            cv2.imwrite(os.path.join(path, 'table_number.jpg'), blackAndWhiteImage)
            
        rospy.loginfo("foto guardada")
        rospy.loginfo(path)

        cv2.drawContours(img, contornos, -1, (255, 255, 255), 3)
        