#! /usr/bin/env python

import rospy

import actionlib
from prediction_action.msg import PredictionAction, PredictionGoal, PredictionResult, PredictionFeedback

import os

import cv2
from cv2 import *
import number_prediction

def predict_number(goal): #funcion a ejecutar al recibir el goal
    rospy.loginfo("action called")
    path = os.getcwd() + '/catkin_ws/src/cookobot-rekognition/images'
    print(path)
    
    image_file = os.path.join(path, 'table_number.jpg')
    
    # Read image
    image = cv2.imread(image_file)
    cv2.imshow("b", image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
    number_predictor = number_prediction.number_prediction()

    prediction = number_predictor.predict(image)
    
    result = PredictionResult() #se construye el mensaje de respuesta
    result.table_number = prediction
    server.set_succeeded(result) #se ha enviado el goal OK
    
rospy.init_node('prediction_action_server')
server = actionlib.SimpleActionServer('prediction', PredictionAction, predict_number, False) # creamos el servidor de la accion
# los parametros son: nombre del servidor, tipo de accion, funcion a ejecutar  variable que posibilita el
# inicio automatico del servidor

rospy.loginfo("Lanzamos el servidor prediction_action_server")
# Getting path
current_directory = os.getcwd()
os.chdir("..")
server.start() # iniciamos el servidor
rospy.spin() # el servidor queda a la espera de recibir goal