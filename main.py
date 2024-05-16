# Research project Msc. Forensic Science
## Luca Nederhorst
## 2023-2024

import os
import io
from PIL import Image
from pillow_heif import register_heif_opener
import csv
from csv import DictWriter
from filters import LowpassFilter
from convert_heic import ConvertImage


# load images and store in directory which is named after the class (in our case insect species)
all_data_folder_path = "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Photos"

# see all species folders in all data folder
species = os.listdir(all_data_folder_path)

# go over photo inside species folder and add to annotation file called csvfile
metadata = []
for specie in species:
    specie_folder_path = os.path.join(all_data_folder_path, specie) 
    specie_folder = os.listdir(specie_folder_path)
    for photo in specie_folder:
            photo_path = os.path.join(specie_folder_path, photo)
            
            # convert Heic photo to JPG
            ConvertImage(photo_path)

            # apply lowpass filter
            lowpass_photo = LowpassFilter(photo_path)


            metadata_photo = {'specie_folder_path': {specie_folder_path} ,'species_id': {photo}}
            metadata.append(metadata_photo)
          

# create csv file and reader object to read CSV file
csv_file = 'metadata_model.csv'
csv_reader = csv.reader(csv_file)

# put metadata in csv file
with open(csv_file, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["specie_folder_path", "species_id"])
    writer.writeheader()
    for item in metadata:
        writer.writerow(item)





            
            

                  
                  
                  


    





    



    




