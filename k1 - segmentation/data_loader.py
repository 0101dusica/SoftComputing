import os
import csv
from PIL import Image

def load_images_from_folder(folder):
    image_names = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            image_names.append(filename)
    return image_names

def load_csv_as_dict(csv_file_path):
    picture_count_dict = {}
    with open(csv_file_path, mode='r') as csvfile:
        csv_reader = csv.DictReader(csvfile)
        for row in csv_reader:
            picture_name = row['picture'].replace('.jpg', '.png')
            picture_count_dict[picture_name] = int(row['toad_boo_bobomb'])
    return picture_count_dict
