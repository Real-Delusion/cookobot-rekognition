#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import os

class Ros2OpenCV_converter():
    
    def __init__(self):
        #self.image_pub = rospy.Publisher("image_topic_2",Image)
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/turtlebot3/camera/image_raw", Image, self.callback)
            
    def callback(self,data):
        try:
            img = self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)
        
        # Getting path
        current_directory = os.getcwd()
        path = current_directory + '/catkin_ws/src/cookobot-rekognition/images/'

        # Converting image from BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Definimos el rango que abarca el color rojo
        lower_color = np.array([100,150,0])
        upper_color = np.array([140,255,255]) 

        # Creating mask
        mask = cv2.inRange(hsv, lower_color, upper_color)
                                
        # Averiguamos el tamanyo de la imagen recuperada
        height, width, channels = img.shape
        
        # Calculating blob centroide
        M = cv2.moments(mask, False)
        try:
            cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
        except ZeroDivisionError:
            cy, cx = height/2, width/2
        
        # Searching contours color objects
        _, contornos, __  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
        for contorno in contornos:
            tam_contorno = cv2.contourArea(contorno)
            tam_mascara = mask.size
            x = (tam_contorno*100)/tam_mascara

            #Crop table number object
            x, y, w, h = cv2.boundingRect(contorno)
            roi = img[y:y + h, x:x + w]

            # Make black and white
            h,s,v = cv2.split(roi)

            # Make binary
            blur = cv2.GaussianBlur(v,(5,5),0)
            ret3,blackAndWhiteImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # Save image
            cv2.imwrite(os.path.join(path , 'table_number.jpg'), blackAndWhiteImage)   
        
        cv2.drawContours(img, contornos, -1, (255,255,255), 3)

        # Showing live image
        cv2.imshow("Imagen", img) 
            
        cv2.waitKey(3)
        
        
def main():
    ic_obj = Ros2OpenCV_converter()
    rospy.init_node("Ros2OpenCV_converter", anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Finalizar captura y conversion de imagen")
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
        
