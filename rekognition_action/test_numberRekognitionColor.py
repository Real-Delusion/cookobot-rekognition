import unittest
import cv2

from src.numberRekognitionColor import *

class TestMetodoProcessImage(unittest.TestCase):
    def prueba(self):
        f = cv2.imread("image_test.jpg")
        self.assertTrue(extract_number(f))

if __name__=="main":
    unittest.main()