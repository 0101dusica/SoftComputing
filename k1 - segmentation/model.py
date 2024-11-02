import cv2 as cv
import numpy as np

# Segment objects in the image and count them based on defined criteria
def segment_and_count_objects(original_image, processed_image):
    
    # Perform closing and opening transformations to remove noise and smooth object edges
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
    processed_image = cv.morphologyEx(processed_image, cv.MORPH_CLOSE, kernel, iterations=6)
    processed_image = cv.morphologyEx(processed_image, cv.MORPH_OPEN, kernel, iterations=7)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))
    processed_image = cv.morphologyEx(processed_image, cv.MORPH_CLOSE, kernel, iterations=5)

    # Find contours in the preprocessed image
    contours, _ = cv.findContours(processed_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Remove contours with small areas
        if cv.contourArea(contour) < 100:
            cv.drawContours(processed_image, [contour], -1, 0, -1)
    
    # Apply distance transform to the cleaned binary image to help separate objects
    processed_image = cv.distanceTransform(processed_image, cv.DIST_L2, 5)
    _, threshold = cv.threshold(processed_image, 0.5 * processed_image.max(), 255, 0)
    threshold = np.uint8(threshold)
    _, markers = cv.connectedComponents(threshold)
    markers = markers + 1  # Increase markers to separate background from foreground
    markers[processed_image == 0] = 0  # Set background marker

    # Use the watershed algorithm to segment objects
    markers = cv.watershed(original_image, markers)
    original_image[markers == -1] = [255, 0, 0]

    # Initialize a counter for segmented objects
    unique_labels = np.unique(markers)
    objects_counted = 0

    # Process each labeled object
    for label in unique_labels:
        if label == -1 or label == 0:  # Skip background and border markers
            continue
        mask = np.uint8(markers == label) * 255
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv.contourArea(contours[0])
            if 10 < area < 5000:
                if area > 1000:  # Heuristic to count larger objects differently
                    objects_counted += 1
                objects_counted += 1
                # Draw bounding box and object info on the original image
                x, y, w, h = cv.boundingRect(contours[0])
                cv.rectangle(original_image, (x, y), (x + w, y + h), (255, 255, 255), 2)
                cv.putText(original_image, f"counted {objects_counted}, {area}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # cv.imshow("Segmented Image", original_image)
    # cv.waitKey(0)

    return objects_counted