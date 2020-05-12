#!/usr/bin/env python3

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# number_prediction.py
# Emilia Rosa van der Heide
# 06/05/20
# This file contains a class that given an image predicts the
# number in that image using a trained TensorFlow model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import os
os.environ["CUDA_VISIBLE_DEVICES"] ='-1'
import numpy as np
from tensorflow.keras.models import load_model
import cv2


class number_prediction:

    # The model we'll be using to predict
    model = None

    # The shape size the model needs
    height = 0
    width = 0
    
    # Data type the model needs
    dtype = None
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # constructor
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def __init__(self):
        
        path = os.getcwd() + '/catkin_ws/src/cookobot-rekognition/TensorFlow/train/number_prediction_model'

        # We load the model
        self.model = load_model(path)
        
        # The shape sizes the model needs
        self.height = self.model.layers[0].input_shape[1]
        self.width = self.model.layers[0].input_shape[2]
        
        # The type the model needs
        self.dtype = self.model.layers[0].dtype


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # prepare_data(image)
    # this method readies the image for the prediction
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def prepare_data(self, image):        
        
        # Make black and white
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Resize image to fit the model's needs
        image = cv2.resize(image, (self.height, self.width), interpolation = cv2.INTER_NEAREST)
        '''
        cv2.imshow("bw", image)
        cv2.waitKey(0)
        '''
        # Change image compression to fit model's needs
        image = image.astype(self.dtype)
        image /= 255.
        
        # Add one dimension to the image
        data = np.expand_dims(image, 0)

        return data
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # predict(image)
    # this method predicts the number in an image
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    def predict(self, image):
        
        # First we need to prepare the image for the model
        data = self.prepare_data(image)
        
        # Now we predict
        predictions = self.model.predict(data)
        
        # We return the prediction with max confidence
        return np.argmax(predictions)

'''
# Load the image
image_file = '../../images/table_number.jpg'
image = cv2.imread(image_file)

number_predictor = number_prediction()

prediction = number_predictor.predict(image)

# The best prediction
print(prediction)
'''
