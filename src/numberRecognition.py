#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class Ros2OpenCV_converter():
    
    def __init__(self):
        #self.image_pub = rospy.Publisher("image_topic_2",Image)
        
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/turtlebot3/camera/image_raw", Image, self.callback)
            
    def callback(self,data):
        try:
            image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

        thresh = cv2.threshold(sharpen,160,255, cv2.THRESH_BINARY_INV)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        min_area = 200000
        max_area = 380000
        for c in cnts:
            area = cv2.contourArea(c)

            if area > min_area and area < max_area:
                x,y,w,h = cv2.boundingRect(c)
                ROI = image[y:y+h, x:x+h]
                cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                cv2.drawContours(image, cnts, -1, (255,255,255), 3) 
         
        
        cv2.imshow("Imagen original", image)
        cv2.imshow("Imagen original", close)

                                          
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
        
