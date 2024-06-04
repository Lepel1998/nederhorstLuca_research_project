
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
from skimage import measure
from skimage.feature import graycomatrix, graycoprops




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


def binary_image(img):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # you need to invert image 
    gray_inverted = cv2.bitwise_not(gray_img)

    # make image fully black and white with cv2.threshold()
    _,binary = cv2.threshold(gray_inverted, 90, 255, cv2.THRESH_BINARY)

    return binary


def geometric_feature(photo_path):
    """ 
        Wen, C., & Guyer, D. (2012). Image-based orchard insect automated 
        identification and classification method. Computers and electronics 
        in agriculture, 89, 110-115.
        Global feature extraction used as this is shown to be working better.
    """

    # load and convert image to HSV 
    photo = cv2.imread(photo_path)
    binary = binary_image(photo)
    #plt.imshow(binary, cmap='gray')
    #plt.show()

    # label the foreground object and exclude background (foreground is 1, background is 0)
    number_labels, labels_binary = cv2.connectedComponents(binary)
    labels_binary = labels_binary.astype(np.uint8)

    # get features from photo for regions
    features = measure.regionprops(labels_binary)

    ## Geometric features (Wen et al., 2009a,b)
    geometric_features_list = []
    for feature in features:
        if feature.label==0:
            continue
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
        #print('Area:', area)

    return geometric_features_list


def fourier(photo_path):
    """
    https://www.youtube.com/watch?v=JfaZNiEbreE&list=PLCeWwpzjQu9gc9C9-iZ9WTFNGhIq4-L1X
    """

    # load image
    img = cv2.imread(photo_path)

    # get binary image
    binary = binary_image(img)

    # show binary image
    #plt.imshow(binary, cmap='gray')

    # detecting the contours in an image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Number of contours found = {format(len(contours))}')

    # read coloured image in for drawing the contours (otherwise the lines will be drawed in white)
    #cv2.drawContours(img, contours, -1, (0,255,0),1)
    #plt.figure(figsize=[10,10])
    #plt.imshow(img[:,:,::-1])
    #plt.axis("off")
    #plt.show()

    # fourier transformation (https://docs.opencv.org/4.x/de/dbc/tutorial_py_fourier_transform.html)
    # contour = max(contours, key=cv2.contourArea)
    contour = max(contours, key=cv2.contourArea)
    fourier_contour = np.fft.fft2(contour)
    fourier_shift = np.fft.fftshift(fourier_contour)
    magnitude_spectrum = 20*np.log(np.abs(fourier_shift))

    magnitude_spectrum_first_two_spat_freq = magnitude_spectrum[:2]

    spat_freq_1 = magnitude_spectrum_first_two_spat_freq[0][0][0]
    spat_freq_2 = magnitude_spectrum_first_two_spat_freq[0][0][1]
    #print(spat_freq_1, spat_freq_2)

    return spat_freq_1, spat_freq_2




def minimum_rectangle_image(photo_path):
    """
        https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    """

    img = cv2.imread(photo_path)
    img = cv2.GaussianBlur(img, (5,5), 0)
    binary = binary_image(img)

    # get contours
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f'amount of contours found {len(contours)}')

    # get coordinates of minimum bounding rectangle
    if len(contours) > 0:
        contour = contours[0]

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        rectangle_image = img[y:y+h, x:x+w]

        # display original image with rectangle
        #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #plt.title('Ímage with bounding Rectangle')
        #plt.show()

        if rectangle_image.size > 0:
            #plt.imshow(rectangle_image)
            #plt.title('Extracted Bouding Rectangle')
            #plt.show()
            pass
        else:
            print('Rectangle image is empty')
    else:
        print('No contours found.')
    return rectangle_image


def invariant_moments(photo_path):
    """
    https://www.tutorialspoint.com/how-to-compute-hu-moments-of-an-image-in-opencv-python
    """
    rect_image = minimum_rectangle_image(photo_path)
    gray_rect_image = cv2.cvtColor(rect_image, cv2.COLOR_BGR2GRAY)

    moments = cv2.moments(gray_rect_image)
    hu_moments = cv2.HuMoments(moments)
    
    return hu_moments[0][0], hu_moments[1][0], hu_moments[2][0], hu_moments[3][0], hu_moments[4][0], hu_moments[5][0], hu_moments[6][0]


def texture(photo_path):
    """
    took avarage of texture features to add it to the csv file
    """

    image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)

    # computer gray level co occurence
    distances=[1]
    angles=[0, np.pi/4,np.pi/2, 3*np.pi/4]
    gray_level_co_occurence = graycomatrix(image, distances, angles)

    # extract texture props of the GLCM
    contrast = graycoprops(gray_level_co_occurence, 'contrast')
    dissimilarity = graycoprops(gray_level_co_occurence, 'dissimilarity')
    homogeneity = graycoprops(gray_level_co_occurence, 'homogeneity')
    energy = graycoprops(gray_level_co_occurence, 'energy')
    correlation = graycoprops(gray_level_co_occurence,'correlation')
    ASM = graycoprops(gray_level_co_occurence, 'ASM')

    mean_contrast = contrast[0].mean(axis=0)
    mean_dissimilarity = dissimilarity[0].mean(axis=0)
    mean_homogeneity = homogeneity[0].mean(axis=0)
    mean_energy = energy[0].mean(axis=0)
    mean_correlation = correlation[0].mean(axis=0)
    mean_ASM = ASM[0].mean(axis=0)

    return mean_contrast, mean_dissimilarity, mean_homogeneity, mean_energy, mean_correlation, mean_ASM
