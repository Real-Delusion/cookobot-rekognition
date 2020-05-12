#! /usr/bin/env python

import rospy

import time
import actionlib
from rekognition_action.msg import RekognitionAction, RekognitionGoal, RekognitionResult, RekognitionFeedback

import numberRekognitionColor
import cv2
from cv2 import *


def rekognize_number(goal): #funcion a ejecutar al recibir el goal
    rospy.loginfo("action called")
    
    # Start finding number block
    obj = numberRekognitionColor.Ros2OpenCV_converter().rekogniceNumber()

    rospy.loginfo("Photo taken and saved")

    result = RekognitionResult() # se construye el mensaje de respuesta
    result.photo = True
    rospy.loginfo(result)
    server.set_succeeded(result) 
    
rospy.init_node('rekognition_action_server')
server = actionlib.SimpleActionServer('rekognition', RekognitionAction, rekognize_number, False) # creamos el servidor de la accion
# los parametros son: nombre del servidor, tipo de accion, funcion a ejecutar  variable que posibilita el
# inicio automatico del servidor

rospy.loginfo("Lanzamos el servidor rekognition_action_server")
server.start() # iniciamos el servidor
rospy.spin() # el servidor queda a la espera de recibir goal