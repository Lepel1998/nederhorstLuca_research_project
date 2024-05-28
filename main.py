

import os
import io
from PIL import Image
from pillow_heif import register_heif_opener
import csv
from csv import DictWriter
from convert_heic import ConvertHeicJpg
from filters import LowpassFilter, HighpassFilter
from augmentate import Augmentation
from ignore_files import IgnoreFiles
import shutil
register_heif_opener()
import cv2
import numpy as np
import re
import shutil




# Research project Msc. Forensic Science
## Luca Nederhorst
## 2023-2024


# load images and store in directory which is named after the class (in our case insect species)
all_data_folder_path = "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Dataset"

# create augmentated directory which has same buildup as original directory
if not os.path.exists("C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Processed_Dataset"):
    shutil.copytree(all_data_folder_path, "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Processed_Dataset", ignore = IgnoreFiles)
         
augment_data_folder_path = "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Processed_Dataset"

# see all species folders in all data folder
species = os.listdir(all_data_folder_path)

# go over photo inside species folder and add to annotation file called csvfile
metadata = []
for specie in species:
    specie_folder_path = os.path.join(all_data_folder_path, specie) 
    specie_folder_path = os.path.normpath(specie_folder_path)

    specie_folder = os.listdir(specie_folder_path)
    
    # convert heic images to jpg images function
    ConvertHeicJpg(specie_folder_path)
    
    
    for photo in specie_folder:
            photo_path = os.path.join(specie_folder_path, photo)
            photo_path = os.path.normpath(photo_path)
            photo_path = photo_path.replace("\\", "\\\\")

            # Lowpass filter to smoothen photo en reduce noise
            lowpass_filtered_photo = LowpassFilter(photo_path, 4)

            # Highpass filter to sharpen filtered photo
            #highpass_filtered_photo = HighpassFilter(photo_path, lowpass_filtered_photo)
            
            # Augmentate pictures
            augmentations = Augmentation(lowpass_filtered_photo)
            augmentation_names = ["resized", "rot90", "rot190", "rot270", "randomcrop", "centercrop", "rot90flip", "rot180flip", "rot270flip"]

            # Save augmented pictures
            for augmentation in range(len(augmentations)):
                 augment_path = os.path.join(augment_data_folder_path, specie, f"{augmentation_names[augmentation]} {photo}" )

                 # save augmented pictures to folder
                 augmentations[augmentation].save(augment_path)

                 # add metadata augmented pictures to annotation file
                 metadata_photo = {'augment_specie_folder_path': {augment_path}, 'species_id': {photo}, 'augmentation': {augmentation_names[augmentation]}}
                 metadata.append(metadata_photo)

# create csv file and reader object to read CSV file
csv_file = 'metadata_model.csv'
csv_reader = csv.reader(csv_file)

# put metadata in csv file
with open(csv_file, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["augment_specie_folder_path", "species_id", "augmentation"])
    writer.writeheader()
    for item in metadata:
        writer.writerow(item)





            
            

                  
                  
                  


    





    



    




