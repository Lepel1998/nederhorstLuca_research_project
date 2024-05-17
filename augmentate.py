# File for functions used in main file

import os
import io
from PIL import Image
from pillow_heif import register_heif_opener
import csv
from csv import DictWriter
from PIL import Image, ImageFilter, ImageChops
import cv2



## augmentation applied (flipping, rotation, and cropping)
## Goal: increae training set to increase accuracy and reduce problems with overfitting


def Augmentation(photo_path):
    photo = Image.open(photo_path)

    # rescaling image to 227x227 pixels
    resized_photo = photo.resize((227, 227))

    # 90 degrees rotation clockwise
    rot90 = resized_photo.rotate(90)

    # 180 degrees rotation clockwise
    rot180 = resized_photo.rotate(180)
    rot180.show()

    # 270 degrees rotation clockwise

    # random crop

    # center crop

    # 90 rotation clockwise with flipping right to left

    # 180 rotation clockwise with flipping right to left

    # 270 rotation with flipping right to left
    return