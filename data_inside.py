import os
import csv
import shutil
from pillow_heif import register_heif_opener  # type: ignore
from functions import augmentation_function, convert_heic_jpg, highpass_filter, ignore_files, lowpass_filter

# register HEIF opener
register_heif_opener()

folder_name = 'dataset'
current_directory = os.getcwd()
DATA_FOLDER_PATH = os.path.join(current_directory, folder_name)
DATA_FOLDER_PATH = os.path.normpath(DATA_FOLDER_PATH).replace("\\", "/")



print(DATA_FOLDER_PATH)
for item in os.listdir(DATA_FOLDER_PATH):
    print(item)




