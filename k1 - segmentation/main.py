import time
import sys
import cv2 as cv
import numpy as np
import os
import csv

# Load all images with the .png extension from the specified dataset folder
def load_images_from_folder(folder):
    return [filename for filename in os.listdir(folder) if filename.endswith(".png")]

# Load data from a CSV file as a dictionary with picture names as keys and object counts as values
# Convert all image file extensions from .jpg to .png in the picture names
def load_csv_as_dict(csv_file_path):
    picture_count_dict = {}
    with open(csv_file_path, mode='r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            picture_name = row['picture'].replace('.jpg', '.png')
            picture_count_dict[picture_name] = int(row['toad_boo_bobomb'])
    return picture_count_dict

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

# Evaluate the model's accuracy by comparing predicted object counts to ground truth values
def evaluate_model(predictions, ground_truth):
    y_true = [ground_truth[image] for image in predictions]
    y_pred = list(predictions.values())
    absolute_errors = list(map(lambda y: abs(y[0] - y[1]), zip(y_true, y_pred)))
    mean_absolute_error = sum(absolute_errors) / len(absolute_errors)
    return mean_absolute_error

# Save the model's prediction results to a text file
def save_results(predictions, ground_truth, mae, elapsed_time, output_file="results.txt"):
    with open(output_file, "w") as f:
        f.write(f"{'Picture':<20} {'Prediction':<15} {'Truth'}\n")
        f.write("-" * 45 + "\n")
        for picture in sorted(predictions.keys()):
            pred = predictions.get(picture, "N/A")
            truth = ground_truth.get(picture, "N/A")
            f.write(f"{picture:<20} {pred:<15} {truth}\n")
        f.write("-" * 45 + "\n")
        f.write(f"\nMean Absolute Error: {mae}\n")
        f.write(f"Elapsed time: {elapsed_time} seconds\n")

# Process images from a specified folder, segment and count objects, and evaluate the model's predictions
def process_and_evaluate(image_folder, ground_truth_path):
    predictions = {}
    image_names = load_images_from_folder(image_folder)
    ground_truth_dict = load_csv_as_dict(ground_truth_path)

    for image_name in image_names:
        image_path = os.path.join(image_folder, image_name)
        original_image, mask = preprocess_image(image_path)
        predictions[image_name] = segment_and_count_objects(original_image, mask)
    
    mae = evaluate_model(predictions, ground_truth_dict)
    return predictions, ground_truth_dict, mae

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    image_folder = sys.argv[1]
    ground_truth_path = os.path.join(image_folder, "object_count.csv")

    start_time = time.time()
    predictions, ground_truth_dict, mae = process_and_evaluate(image_folder, ground_truth_path)
    elapsed_time = round(time.time() - start_time, 2)

    save_results(predictions, ground_truth_dict, mae, elapsed_time)
    print("Mean Absolute Error: ",mae)