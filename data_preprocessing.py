# Research project Msc. Forensic Science
## Luca Nederhorst
## 2023-2024

import os
import io
from PIL import Image
from pillow_heif import register_heif_opener
import csv
from csv import DictWriter


# load images and store in directory which is named after the class (in our case insect species)
all_data_folder_path = "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Photos"

# see all species folders in all data folder
species = os.listdir(all_data_folder_path)

# go over photo inside species folder and add to annotation file called csvfile
metadata = []
for specie in species:
    image_path = os.path.join(all_data_folder_path, specie)
    #print(f"Contents of folder: {specie_path}")
    specie_folder = os.listdir(image_path)
    for photo in specie_folder:
            metadata_photo = {'image_path': {image_path} ,'species_id': {photo}}
            metadata.append(metadata_photo)
          

# create csv file and reader object to read CSV file
csv_file = 'metadata_model.csv'
csv_reader = csv.reader(csv_file)

# put metadata in csv file
with open(csv_file, 'a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=["image_path", "species_id"])
    writer.writeheader()
    for item in metadata:
        writer.writerow(item)





            
            

                  
                  
                  


    





    



    




