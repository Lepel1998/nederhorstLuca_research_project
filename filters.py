# File for functions used in main file

import os
import io
from PIL import Image
from pillow_heif import register_heif_opener
import csv
from csv import DictWriter
from PIL import Image, ImageFilter, ImageChops
import cv2
import numpy as np


register_heif_opener()


# in this part of the code, the function will be used to enhance the image imported. Based on article by Makander & Halalli (2015), Gaussian filter works the best for both lowpass and highpass filters
## lowpass filter is to smooth the image to reduce noise and minor details
def LowpassFilter(photo_path, radius):
    photo = Image.open(photo_path)
    lowpass_filtered_photo = photo.filter(ImageFilter.GaussianBlur(radius))
    return lowpass_filtered_photo

## highpass filter is to sharpen the picture
def HighpassFilter(photo_path, lowpass_filtered_photo):
    photo = Image.open(photo_path)
    highpass_filtered_photo = ImageChops.subtract(photo, lowpass_filtered_photo, scale = 1, offset = 2)
    return highpass_filtered_photo





