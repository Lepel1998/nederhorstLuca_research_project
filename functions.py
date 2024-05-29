
"""
Module: functions.py

This module contains functions for image processing and augmentation.

"""

import os
import shutil
from random import randrange
from PIL import Image, ImageFilter, ImageChops  # type: ignore


def augmentation_function(photo):
    """ Augmentation function to flip, rotate and crop images """

    resized_photo = photo.resize((227, 227))
    rot90 = resized_photo.rotate(90)
    rot180 = resized_photo.rotate(180)
    rot270 = resized_photo.rotate(270)

    crop = 150
    # random crop
    width, height = resized_photo.size
    width1 = randrange(0, width - crop)
    height1 = randrange(0, height - crop)
    randomcrop = resized_photo.crop((width1,
                                     height1,
                                     width1 + crop,
                                     height1 + crop
                                     ))

    # center crop
    right_border_image = (width + crop)/2
    bottom_border_image = (height + crop)/2
    left_border_image = (width - crop)/2
    top_border_image = (height - crop)/2
    centercrop = resized_photo.crop((left_border_image,
                                     top_border_image,
                                     right_border_image,
                                     bottom_border_image))

    # 90 rotation clockwise with flipping right to left
    rot90flip = rot90.transpose(method=Image.FLIP_LEFT_RIGHT)

    # 180 rotation clockwise with flipping right to left
    rot180flip = rot180.transpose(method=Image.FLIP_LEFT_RIGHT)

    # 270 rotation with flipping right to left
    rot270flip = rot270.transpose(method=Image.FLIP_LEFT_RIGHT)

    return resized_photo, rot90, rot180, rot270, randomcrop, centercrop, rot90flip, rot180flip, rot270flip


def convert_heic_jpg(heic_folder):
    """ Convert heic to jpg function """
    for heic_image in os.listdir(heic_folder):
        if heic_image.lower().endswith('.heic'):
            # create paths for the HEIC and JPG files
            heic_file_path = os.path.join(heic_folder, heic_image)
            jpg_file_path = os.path.join(heic_folder,
                                         heic_image.replace('.heic', '.jpg'))

            # replace the HEIC file with the JPG file
            shutil.move(heic_file_path, jpg_file_path)
            print(f"Replaced {heic_image} with {heic_image.replace('.heic', '.jpg')}")


def highpass_filter(photo_path, lowpass_filtered_photo):
    """ Highpass Gaussian filter to sharpen image (Makander & Halalli (2015) """
    photo = Image.open(photo_path)
    highpass_filtered_photo = ImageChops.subtract(photo,
                                                  lowpass_filtered_photo,
                                                  scale=1,
                                                  offset=2)
    return highpass_filtered_photo


def ignore_files(directory, files):
    """" Ignore files in folder """
    return [f for f in files if os.path.isfile(os.path.join(directory, f))]


def lowpass_filter(photo_path, radius):
    """ Lowpass Gaussian filter to smoothen image/reduce noisee (Makander & Halalli (2015) """
    photo = Image.open(photo_path)
    lowpass_filtered_photo = photo.filter(ImageFilter.GaussianBlur(radius))
    return lowpass_filtered_photo
