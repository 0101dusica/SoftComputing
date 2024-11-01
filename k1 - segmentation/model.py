import cv2 as cv
import numpy as np

def segmentation(original_image, preprocessed_image):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    preprocessed_image = cv.morphologyEx(preprocessed_image, cv.MORPH_CLOSE, kernel, iterations=2)
    preprocessed_image = cv.morphologyEx(preprocessed_image, cv.MORPH_OPEN, kernel, iterations=5)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (4,4))
    preprocessed_image = cv.morphologyEx(preprocessed_image, cv.MORPH_CLOSE, kernel, iterations=6)


    distTransform = cv.distanceTransform(preprocessed_image, cv.DIST_L2, 5)
    _, distThreshold = cv.threshold(distTransform, 15, 255, cv.THRESH_BINARY)

    distThreshold = np.uint8(distThreshold)
    _, labels = cv.connectedComponents(distThreshold)

    labels = np.int32(labels)
    labels = cv.watershed(original_image, labels)

    original_image[labels == -1] = [255,0,0]
    
    unique_colours = np.unique(labels)
    objects_counted = len(unique_colours) - 2

    # Visualize the segmentation results
    # cv.imshow("Segmented Image", original_image)
    # cv.waitKey(0)

    return objects_counted