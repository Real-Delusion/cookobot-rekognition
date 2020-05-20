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
            cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)

        '''
        # Convertimos la imagen recortada de BGR a HSV
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Definimos el rango que abarca el color rojo
        lower_color = np.array([0,186,100])
        upper_color = np.array([25,255,255]) 

        # Creamos una mascara para filtrar solo los pixeles rojos
        # que hay en la imagen capturada
        mask = cv2.inRange(hsv, lower_color, upper_color)
                                
        # Averiguamos el tamanyo de la imagen recuperada
        height, width, channels = cv_image.shape
        
        # Calculamos el centroide del blob de la imagen binaria utilizando
        # los momentos de la imagen
        M = cv2.moments(mask, False)
        try:
            cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
        except ZeroDivisionError:
            cy, cx = height/2, width/2
        
        # Buscamos los contornos de los objetos rojos de la imagen capturada
        _, contornos, __  = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    
        for contorno in contornos:
		tam_contorno = cv2.contourArea(contorno)
		tam_mascara = mask.size
		x = (tam_contorno*100)/tam_mascara
            	if (x > 1):
                	# Escribimos en la imagen el instante en el que identificamos 
                	# el objeto rojo
                	cv2.putText(cv_image, 
                                "Blob rojo localizado", 
                                (int(cx),int(cy)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)     
        
        cv2.drawContours(cv_image, contornos, -1, (255,255,255), 3)                   
        
        cv2.imshow("Imagen original", cv_image)
                                          
        cv2.waitKey(3)
        '''
        
        
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
        
