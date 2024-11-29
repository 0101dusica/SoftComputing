import cv2
import os
import sys
from sklearn.metrics import mean_absolute_error

# Load ground truth counts from a CSV file
def load_ground_truth_counts(csv_file_path):
    counts = {}
    with open(csv_file_path, 'r') as file:
        next(file)
        for line in file:
            video_name, count = line.strip().split(',')
            counts[video_name] = int(count)
    return counts

# Detect moving objects by comparing three consecutive frames and return detected objects and binary mask
def detect_motion(prev_frame, current_frame, next_frame, min_area=200, max_area=1000):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    diff_prev_curr = cv2.absdiff(prev_gray, curr_gray)
    diff_curr_next = cv2.absdiff(curr_gray, next_gray)

    motion_mask = cv2.bitwise_and(diff_prev_curr, diff_curr_next)

    _, binary_mask = cv2.threshold(motion_mask, 30, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_objects = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(contour)
            centroid = (x + w // 2, y + h // 2)
            detected_objects.append({"area": area, "width_height": (w, h), "centroid": centroid})
            
    return detected_objects, binary_mask

# Process video to detect moving objects and count the number of objects crossing a defined line
# Source: Based on ideas from https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
def count_objects_in_video(video_path, line_y_ratio=0.7, min_area=200, max_area=1000):
    cap = cv2.VideoCapture(video_path)
    object_count = 0
    processed_centroids = set()

    ret, prev_frame = cap.read()
    if not ret:
        return object_count

    ret, current_frame = cap.read()
    if not ret:
        return object_count

    ret, next_frame = cap.read()
    if not ret:
        return object_count

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_y_position = int(frame_height * line_y_ratio)

    while ret:
        detected_objects, _ = detect_motion(prev_frame, current_frame, next_frame, min_area, max_area)

        for obj in detected_objects:
            centroid = obj["centroid"]
            w, h = obj["width_height"]

            line_tolerance = 1
            if (line_y_position - line_tolerance <= centroid[1] <= line_y_position + line_tolerance 
                    and tuple(centroid) not in processed_centroids 
                    and (0.5 <= w / h < 5.0) and (25 < w < 47) and (25 < h < 40)):
                object_count += 1
                processed_centroids.add(tuple(centroid))

        prev_frame = current_frame
        current_frame = next_frame
        ret, next_frame = cap.read()

    cap.release()
    return object_count

# Evaluate predictions using Mean Absolute Error between predicted and ground truth values
# Source: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html
def calculate_mean_absolute_error(predictions, ground_truth):
    y_true = [ground_truth[video] for video in predictions]
    y_pred = [predictions[video] for video in predictions]
    return mean_absolute_error(y_true, y_pred)

# Main function to process videos and evaluate predictions
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    data_directory = sys.argv[1]
    counts_file_path = os.path.join(data_directory, 'counts.csv')
    ground_truth_counts = load_ground_truth_counts(counts_file_path)

    predictions = {}
    for video_file in os.listdir(data_directory):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(data_directory, video_file)
            predictions[video_file] = count_objects_in_video(video_path)

    mae = calculate_mean_absolute_error(predictions, ground_truth_counts)
    print(mae)
