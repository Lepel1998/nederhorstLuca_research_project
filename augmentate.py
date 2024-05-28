# File for functions used in main file

import os
import io
from PIL import Image
from pillow_heif import register_heif_opener
import csv
from csv import DictWriter
from PIL import Image, ImageFilter, ImageChops
import cv2
from random import randrange






## augmentation applied (flipping, rotation, and cropping)
## Goal: increase training set to increase accuracy and reduce problems with overfitting


def Augmentation(photo):
    #photo = Image.open(photo_path)

    # rescaling image to 227x227 pixels
    resized_photo = photo.resize((227, 227))

    # 90 degrees rotation clockwise
    rot90 = resized_photo.rotate(90)

    # 180 degrees rotation clockwise
    rot180 = resized_photo.rotate(180)

    # 270 degrees rotation clockwise
    rot270 = resized_photo.rotate(270)
    
    # random crop
    width, height = resized_photo.size
    crop = 150
    width1 = randrange(0, width - crop)
    height1 = randrange (0, height - crop)
    randomcrop = resized_photo.crop((width1,height1,width1+crop,height1+crop))

    # center crop
    cropsize = 130
    right_border_image = (width + cropsize)/2
    bottom_border_image = (height + cropsize)/2
    left_border_image = (width - cropsize)/2
    top_border_image = (height - cropsize)/2
    centercrop = resized_photo.crop((left_border_image, top_border_image, right_border_image, bottom_border_image))
    
    # 90 rotation clockwise with flipping right to left
    rot90flip = rot90.transpose(method=Image.FLIP_LEFT_RIGHT)

    # 180 rotation clockwise with flipping right to left
    rot180flip = rot180.transpose(method=Image.FLIP_LEFT_RIGHT)

    # 270 rotation with flipping right to left
    rot270flip = rot270.transpose(method=Image.FLIP_LEFT_RIGHT)

    return resized_photo, rot90, rot180, rot270, randomcrop, centercrop, rot90flip, rot180flip, rot270flip