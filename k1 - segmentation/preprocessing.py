import cv2 as cv
import numpy as np

def load_image(image_path):    
    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    height, _ = image.shape[:2]
    image = image[:int(height * 0.75), :]

    return image

def preprocess_image(image_path):
    original_image = load_image(image_path)

    preprocessed_image = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
    
    lower = np.array([13,13,43])
    upper = np.array([109, 255, 255])
    
    preprocessed_image = cv.inRange(preprocessed_image, lower, upper)

    _, preprocessed_image = cv.threshold(preprocessed_image, 50, 255, cv.THRESH_BINARY_INV)

    return original_image, preprocessed_image