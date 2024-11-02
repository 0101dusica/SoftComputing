from data_loader import *
from preprocessing import *
from model import *
from evaluate import *
import time
import sys

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