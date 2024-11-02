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