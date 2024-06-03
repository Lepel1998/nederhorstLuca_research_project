
"""
Module: functions.py

This module contains functions for image processing and augmentation.

"""

import os
import shutil
from random import randrange
from PIL import Image, ImageFilter, ImageChops, ImageOps  # type: ignore
import cv2
import numpy as np

import sklearn
from sklearn.cluster import KMeans
from scipy.ndimage import binary_opening
import skimage as ski
from skimage import io
from skimage.color import rgb2hsv, rgb2gray, rgb2lab
#from skimage.draw import polygen_perimeter

import matplotlib.pyplot as plt
from skimage.measure import regionprops, find_contours
#from skimage.feature import greycomatrix, greycoprops
from skimage.transform import resize



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


def geometric_feature(photo_path):
    """ 
        Wen, C., & Guyer, D. (2012). Image-based orchard insect automated 
        identification and classification method. Computers and electronics 
        in agriculture, 89, 110-115.
        Global feature extraction used as this is shown to be working better.
    """

    # load and convert image to HSV 
    photo = cv2.imread(photo_path)
    hsv_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)

    # K-means clustering to segment the insect
    # the variable hue is a column vector containing the hue values of all pixels in the image
    # each element in this vector represents the hue value of a pixel
    # there are two labels, the background and object (insect)
    hue = hsv_photo[:,:,0].reshape(-1,1)
    kmeans = KMeans(n_clusters=3)
    kmeans = kmeans.fit(hue)
    labels = kmeans.labels_.reshape(hsv_photo.shape[:2])

    # label the clusters (foreground and background), binary photo is the photo consist of foreground and background
    if np.sum(labels==0) < np.sum(labels==1):
        insect_label=0
    else:
        insect_label=1
    
    # create binary image where pixels corresponding to the insect are set to 1
    binary_photo = np.zeros_like(labels, dtype=np.uint8)
    binary_photo[labels==insect_label]=1
   
    # clean the image from noise
    cleaned_binary_photo = binary_opening(binary_photo, structure=np.ones((3,3))).astype(np.uint8)

    # get features from photo
    ## Geometric features (Wen et al., 2009a,b)
    geometric_features_list = []

    for feature in regionprops(cleaned_binary_photo.astype(int)):
        area = feature.area
        geometric_features_list.append(feature.area)

        perimeter = feature.perimeter
        geometric_features_list.append(feature.perimeter)

        circularity_ratio = (4*np.pi*area/perimeter**2) 
        geometric_features_list.append(circularity_ratio) 

        geometric_features_list.append(feature.eccentricity)
        geometric_features_list.append(feature.major_axis_length)

        geometric_features_list.append(feature.minor_axis_length)
        geometric_features_list.append(feature.convex_area)
        geometric_features_list.append(feature.solidity)
        geometric_features_list.append(feature.equivalent_diameter_area)
   
    return geometric_features_list