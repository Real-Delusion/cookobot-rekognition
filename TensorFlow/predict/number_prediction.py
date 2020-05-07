import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load the image
image_file = 'number.jpg'
image = cv2.imread(image_file)

# Make back and white
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize image to fit the model's needs
image = cv2.resize(image, (28, 28))

# Change image compression to fit model's needs
image = image.astype(np.float32)
image /= 255.

# Add one dimension to the image
data = np.expand_dims(image, 0)

# We import our previously trained model
model = load_model('../train/number_prediction_model')

# Predict the number of the data
predictions = model.predict(data)

# The best prediction
print(np.argmax(predictions))
