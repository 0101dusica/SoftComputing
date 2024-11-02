import cv2 as cv
import numpy as np

# Load an image and crop out irrelevant parts from the top and bottom
def load_image(image_path):    
    image = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    height, _ = image.shape[:2]
    top_crop, bottom_crop = int(height * 0.15), int(height * 0.25)
    return image[top_crop:height - bottom_crop, :]

# Prepare the image for segmentation
# Apply a color mask to isolate specific color ranges (HSV lower and upper bounds found using this tool: https://pseudopencv.site/utilities/hsvcolormask/)
# Threshold the masked image to create a binary inverse for segmentation
def preprocess_image(image_path):
    original_image = load_image(image_path)
    preprocessed_image = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)
    lower, upper = np.array([13, 13, 43]), np.array([109, 255, 255])
    preprocessed_image = cv.inRange(preprocessed_image, lower, upper)
    _, preprocessed_image = cv.threshold(preprocessed_image, 50, 255, cv.THRESH_BINARY_INV)
    return original_image, preprocessed_image