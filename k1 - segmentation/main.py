from data_loader import *
from preprocessing import *
from model import *
from evaluate import *
import time

def print_results(predictions, ground_truth):
    print(f"{'Picture':<20} {'Prediction':<15} {'Truth'}")
    print("-" * 45)

    for picture in sorted(predictions.keys()):
        pred = predictions.get(picture, "N/A")
        truth = ground_truth.get(picture, "N/A")
        print(f"{picture:<20} {pred:<15} {truth}")
    print("-" * 45)

def process_and_evaluate(image_folder, ground_truth_path):
    predictions = {}
    
    image_names = load_images_from_folder(image_folder)
    ground_truth_dict = load_csv_as_dict(ground_truth_path)
    
    for image_name in image_names:
        image_path = os.path.join(image_folder, image_name)
        
        original_image, preprocessed_image = preprocess_image(image_path)
        total_objects = segmentation(original_image, preprocessed_image)
        
        predictions[image_name] = total_objects
    
    print_results(predictions, ground_truth_dict)
    mae = evaluate_model(predictions, ground_truth_dict)
    return mae


if __name__ == "__main__":
    image_folder = "data/"
    ground_truth_path = "data/object_count.csv"

    start_time = time.time()
    mae = process_and_evaluate(image_folder, ground_truth_path)
    end_time = time.time()

    print("Mean Absolute Error:", mae)
    elapsed_time = round(end_time - start_time, 2)
    print(f"Elapsed time: {elapsed_time} seconds")