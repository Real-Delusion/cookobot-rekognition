#! /usr/bin/env python
import rospy

import time
import actionlib
from rekognition_action.msg import RekognitionAction, RekognitionGoal, RekognitionResult, RekognitionFeedback

import numberRekognitionColor
import capturarImagenObjetoRojo
import cv2
from cv2 import *
import threading
from sensor_msgs.msg import Image
import os

def rekognize_number(goal): #funcion a ejecutar al recibir el goal
    rospy.loginfo("action called")
    
    camera_data = rospy.wait_for_message("/turtlebot3/camera/image_raw", Image)

    # Start finding number block
    photoTaken = obj.process_image(camera_data)

    rospy.loginfo("Photo taken and saved")

    result = RekognitionResult() # se construye el mensaje de respuesta
    result.photo = photoTaken
    rospy.loginfo(result)
    server.set_succeeded(result) 
    
# changing path
os.chdir("..")   
rospy.init_node('rekognition_action_server')
server = actionlib.SimpleActionServer('rekognition', RekognitionAction, rekognize_number, False) # creamos el servidor de la accion
# los parametros son: nombre del servidor, tipo de accion, funcion a ejecutar  variable que posibilita el
# inicio automatico del servidor

obj = numberRekognitionColor.Ros2OpenCV_converter()
rospy.loginfo("Lanzamos el servidor rekognition_action_server")
server.start() # iniciamos el servidor
rospy.spin() # el servidor queda a la espera de recibir goal