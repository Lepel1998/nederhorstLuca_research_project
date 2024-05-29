"""
Research project for Msc. Forensic Science

Author: Luca Nederhorst
Academic year: 2023 - 2024
"""

import os
import csv
import shutil
from pillow_heif import register_heif_opener  # type: ignore
from functions import augmentation_function, convert_heic_jpg, highpass_filter, ignore_files, lowpass_filter

# register HEIF opener
register_heif_opener()

# Research project Msc. Forensic Science
# Luca Nederhorst
# 2023-2024


# load images and store in directory which is named after the class
ALL_DATA_FOLDER_PATH = "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Dataset"

# create augmentated directory which has same buildup as original directory
if not os.path.exists("C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Processed_Dataset"):
    shutil.copytree(ALL_DATA_FOLDER_PATH, "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Processed_Dataset", ignore=ignore_files)
else:
    for folder in os.listdir("C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Processed_Dataset"):
        folder_path = os.path.join("C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Processed_Dataset", folder)
        folder_path = os.path.normpath(folder_path)
        folder = os.listdir(folder_path)
        for photo in folder:
            photo_path = os.path.join(folder_path, photo)
            os.remove(photo_path)

AUGMENT_DATA_FOLDER_PATH = "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Processed_Dataset"

# see all species folders in all data folder
species = os.listdir(ALL_DATA_FOLDER_PATH)

# go over photo inside species folder and add to annotation file called csvfile
metadata = []
for specie in species:
    specie_folder_path = os.path.join(ALL_DATA_FOLDER_PATH, specie)
    specie_folder_path = os.path.normpath(specie_folder_path)

    specie_folder = os.listdir(specie_folder_path)

    """ convert heic images to jpg images function """
    convert_heic_jpg(specie_folder_path)

    for photo in specie_folder:
        photo_path = os.path.join(specie_folder_path, photo)
        photo_path = os.path.normpath(photo_path)
        photo_path = photo_path.replace("\\", "\\\\")

        """ Lowpass filter to smoothen photo en reduce noise """
        lowpass_filtered_photo = lowpass_filter(photo_path, 4)

        """ Highpass filter to sharpen filtered photo """
        highpass_filtered_photo = highpass_filter(photo_path,
                                                  lowpass_filtered_photo)

        """ Augmentate pictures """
        augmentations = augmentation_function(highpass_filtered_photo)
        augmentation_names = ["resized",
                              "rot90",
                              "rot190",
                              "rot270",
                              "randomcrop",
                              "centercrop",
                              "rot90flip",
                              "rot180flip",
                              "rot270flip"]

        """ Save augmented pictures and add data to annotation file"""
        for augmentation, augment_data in enumerate(augmentations):
            augment_path = os.path.join(AUGMENT_DATA_FOLDER_PATH,
                                        specie,
                                        f"{augmentation_names[augmentation]} {photo}")
            augment_data.save(augment_path)

            metadata_photo = {'augment_specie_folder_path': {augment_path},
                              'species_id': {photo},
                              'augmentation': {augmentation_names[augmentation]}}
            metadata.append(metadata_photo)

# create csv file and reader object to read CSV file
CSV_FILE = 'metadata_model.csv'

# put metadata in csv file
with open(CSV_FILE, 'a', newline='', encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["augment_specie_folder_path",
                                              "species_id",
                                              "augmentation"])
    writer.writeheader()
    for item in metadata:
        writer.writerow(item)
