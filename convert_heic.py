# Convert heic image to jpg image
import os
import io
from PIL import Image
from pillow_heif import register_heif_opener
import csv
from csv import DictWriter
import shutil

register_heif_opener()


def ConvertHeicJpg(heic_folder):
    # Iterate through files in the folder
    for heic_image in os.listdir(heic_folder):
        if heic_image.lower().endswith('.heic'):
            # Construct paths for the HEIC and JPG files
            heic_file_path = os.path.join(heic_folder, heic_image)
            jpg_file_path = os.path.join(heic_folder, heic_image.replace('.heic', '.jpg'))
       
            # Replace the HEIC file with the JPG file
            shutil.move(heic_file_path, jpg_file_path )
            print(f"Replaced {heic_image} with {heic_image.replace('.heic', '.jpg')}")





    