"""
Research project for Msc. Forensic Science

Author: Luca Nederhorst
Academic year: 2023 - 2024
"""

import os
import csv
import shutil
from pillow_heif import register_heif_opener
from functions import (augmentation_function,
                       convert_heic_jpg,
                       geometric_feature,
                       highpass_filter,
                       ignore_files,
                       lowpass_filter,
                       fourier,
                       invariant_moments,
                       texture,
                       color)

# register HEIF opener
register_heif_opener()

# load photos and store in directory which is named after the class
current_directory = os.getcwd()
DATA_FOLDER_PATH = os.path.join(current_directory, 'final_dataset')
DATA_FOLDER_PATH = os.path.normpath(DATA_FOLDER_PATH).replace("\\", "/")

# create augmentated directory which has same buildup as original directory
if not os.path.exists(os.path.join(current_directory, 'processed_dataset')):
    shutil.copytree(DATA_FOLDER_PATH,
                    os.path.join(current_directory, 'processed_dataset'),
                    ignore=ignore_files)
else:
    for folder in os.listdir(os.path.join(current_directory, 'processed_dataset')):
        folder_path = os.path.join(os.path.join(current_directory, 'processed_dataset'), folder)
        folder_path = os.path.normpath(folder_path)
        folder = os.listdir(folder_path)
        for photo in folder:
            photo_path = os.path.join(folder_path, photo)
            os.remove(photo_path)

AUGMENT_DATA_FOLDER_PATH = os.path.join(current_directory, 'processed_dataset')

# see all species folders in all data folder
species = os.listdir(DATA_FOLDER_PATH)
for heic_folder in species:
    specie_folder_path = os.path.join(DATA_FOLDER_PATH, heic_folder)
    specie_folder_path = os.path.normpath(specie_folder_path)
    convert_heic_jpg(specie_folder_path)


# go over photo inside species folder and add to annotation file called csvfile
metadata = []
for specie in species:
    specie_folder_path = os.path.join(DATA_FOLDER_PATH, specie)
    specie_folder_path = os.path.normpath(specie_folder_path)

    specie_folder = os.listdir(specie_folder_path)

    """ convert heic photo to jpg photo function """
    

    for photo in specie_folder:
        photo_path = os.path.join(specie_folder_path, photo)
        print(photo_path)
        photo_path = os.path.normpath(photo_path)
        photo_path = photo_path.replace("\\", "\\\\")

        """ Lowpass filter to smoothen photo en reduce noise """
        lowpass_filtered_photo = lowpass_filter(photo_path, 5)

        """ Highpass filter to sharpen filtered photo """
        # highpass_filtered_photo = highpass_filter(photo_path, lowpass_filtered_photo)

        """ Augmentate pictures """
        augmentations = augmentation_function(lowpass_filtered_photo)
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

            """ feature extraction of augmented and blurred pictures """
            geometric_features = geometric_feature(augment_path)
            spatial_frequencies = fourier(augment_path)
            hu_moments = invariant_moments(augment_path)
            texture_features = texture(augment_path)
            color_features = color(augment_path)

            # put metadata of photo in file
            metadata_photo = {'augment_specie_folder_path': {augment_path},
                              'species_id': {photo},
                              'augmentation': {augmentation_names[augmentation]},
                              'area': {geometric_features[0]},
                              'perimeter': {geometric_features[1]},
                              'circularity_ratio': {geometric_features[2]},
                              'eccentricity': {geometric_features[3]},
                              'major_axis_length': {geometric_features[4]},
                              'minor_axis_length': {geometric_features[5]},
                              'convex_area': {geometric_features[6]},
                              'solidity': {geometric_features[7]},
                              'equivalent_diameter_area': {geometric_features[8]},
                              'spatial_frequency_1': {spatial_frequencies[0]},
                              'spatial_frequency_2': {spatial_frequencies[1]},
                              'hu_moment_1': {hu_moments[0]},
                              'hu_moment_2': {hu_moments[1]},
                              'hu_moment_3': {hu_moments[2]},
                              'hu_moment_4': {hu_moments[3]},
                              'hu_moment_5': {hu_moments[4]},
                              'hu_moment_6': {hu_moments[5]},
                              'hu_moment_7': {hu_moments[6]},
                              'contrast': {texture_features[0]},
                              'dissimilarity': {texture_features[1]},
                              'homogeneity': {texture_features[2]},
                              'energy': {texture_features[3]},
                              'correlation': {texture_features[4]},
                              'ASM': {texture_features[5]},
                              'mean_hue_hsv': {color_features[0]},
                              'std_hue_hsv': {color_features[1]},
                              'mean_sat_hsv': {color_features[2]},
                              'std_sat_hsv': {color_features[3]},
                              'mean_hue_LCH': {color_features[4]},
                              'std_hue_LCH': {color_features[5]},
                              'mean_sat_LCH': {color_features[6]},
                              'std_sat_LCH': {color_features[7]},
                              'mean_luminance': {color_features[8]},
                              'std_sat_lab': {color_features[9]}}
            metadata.append(metadata_photo)

# create csv file and reader object to read CSV file
CSV_FILE = 'metadata_model.csv'

# put metadata in csv file
with open(CSV_FILE, 'a', newline='', encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["augment_specie_folder_path",
                                              "species_id",
                                              "augmentation",
                                              'area',
                                              'perimeter',
                                              'circularity_ratio',
                                              'eccentricity',
                                              'major_axis_length',
                                              'minor_axis_length',
                                              'convex_area',
                                              'solidity',
                                              'equivalent_diameter_area',
                                              'spatial_frequency_1',
                                              'spatial_frequency_2',
                                              'hu_moment_1',
                                              'hu_moment_2',
                                              'hu_moment_3',
                                              'hu_moment_4',
                                              'hu_moment_5',
                                              'hu_moment_6',
                                              'hu_moment_7',
                                              'contrast',
                                              'dissimilarity',
                                              'homogeneity',
                                              'energy',
                                              'correlation',
                                              'ASM',
                                              'mean_hue_hsv',
                                              'std_hue_hsv',
                                              'mean_sat_hsv',
                                              'std_sat_hsv',
                                              'mean_hue_LCH',
                                              'std_hue_LCH',
                                              'mean_sat_LCH',
                                              'std_sat_LCH',
                                              'mean_luminance',
                                              'std_sat_lab',
                                              ])
    writer.writeheader()
    for item in metadata:
        writer.writerow(item)
