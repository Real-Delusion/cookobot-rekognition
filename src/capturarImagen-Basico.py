#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


class Ros2OpenCV_image_converter(object):

    def __init__(self):
    
        self.bridge_object = CvBridge()
        self.image_sub = rospy.Subscriber("/turtlebot3/camera/image_raw",Image,self.camera_callback)

    def camera_callback(self,data):

        try:
            # Seleccionamos bgr8 porque es la codificacion de OpenCV por defecto
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        cv2.imshow("Imagen capturada por el robot", cv_image)
        (ancho, alto, canales) = cv_image.shape
        rospy.loginfo("- ancho: {}".format(ancho))
        rospy.loginfo("- alto: {}".format(alto))
        rospy.loginfo("- canales: {}".format(canales))
        
        rospy.loginfo("- size: {}".format(cv_image.size))
        rospy.loginfo("- ndim: {}".format(cv_image.ndim))
        rospy.loginfo("- dtype: {}".format(cv_image.dtype))
        rospy.loginfo("- itemsize: {}".format(cv_image.itemsize))
        
        cv2.waitKey(1)    

def main():
    img_converter_object = Ros2OpenCV_image_converter()
    rospy.init_node('Ros2OpenCV_image_converter', anonymous=True)
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Fin del programa!")
    
    cv2.destroyAllWindows() 
    

if __name__ == '__main__':
    main()
