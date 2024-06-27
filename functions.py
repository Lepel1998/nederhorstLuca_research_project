"""
Module: All functions used in main.py file

This module contains functions for photo preprocessing and feature extraction.
These functions are used in main.py.
"""

import os
from random import randrange

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageChops
from skimage import measure
from skimage.feature import graycomatrix, graycoprops
import pillow_heif

pillow_heif.register_heif_opener()


""" Preprocessing of data functions """


def augmentation_function(photo):
    """ 
    Augmentation function to flip, rotate and crop photos 
    (Kanisathan et al. 2021; Mikołajczyk & Grochowski, 2018; Shorten & Khoshgoftaar, 2019)
    """

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
    right_border_photo = (width + crop)/2
    bottom_border_photo = (height + crop)/2
    left_border_photo = (width - crop)/2
    top_border_photo = (height - crop)/2
    centercrop = resized_photo.crop((left_border_photo,
                                     top_border_photo,
                                     right_border_photo,
                                     bottom_border_photo))

    # 90 rotation clockwise with flipping right to left
    rot90flip = rot90.transpose(method=Image.FLIP_LEFT_RIGHT)

    # 180 rotation clockwise with flipping right to left
    rot180flip = rot180.transpose(method=Image.FLIP_LEFT_RIGHT)

    # 270 rotation with flipping right to left
    rot270flip = rot270.transpose(method=Image.FLIP_LEFT_RIGHT)

    return (resized_photo, rot90, rot180,
            rot270, randomcrop, centercrop,
            rot90flip, rot180flip, rot270flip)


def convert_heic_jpg(heic_folder):
    """ Convert heic to jpg function """

    for heic_photo in os.listdir(heic_folder):
        if heic_photo.lower().endswith('.heic'):
            heic_file_path = os.path.join(heic_folder, heic_photo)
            jpg_file_path = os.path.join(heic_folder,
                                         heic_photo.replace('.HEIC', '.jpg'))
            try:
                photo = Image.open(heic_file_path)
                photo.save(jpg_file_path, format='JPEG')
                os.remove(heic_file_path)
                print('Conversion successfull', heic_photo)
            except Exception:
                print('Error converting', heic_photo)
        else:
            print('No conversion needed')


def ignore_files(directory, files):
    """" Ignore files in folder """

    return [f for f in files if os.path.isfile(os.path.join(directory, f))]


def lowpass_filter(photo_path, radius):
    """ 
    Lowpass Gaussian to smoothen photo/reduce noise 
    (Makander & Halalli, 2015)
    """

    photo = Image.open(photo_path)
    lowpass_filtered_photo = photo.filter(ImageFilter.GaussianBlur(radius))

    return lowpass_filtered_photo


""" Extract features functions """


def binary_photo(photo):
    """ Converts photo to binary photo """

    gray_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    gray_inverted = cv2.bitwise_not(gray_photo)
    _, binary = cv2.threshold(gray_inverted, 90, 255, cv2.THRESH_BINARY)

    return binary


def geometric_feature(photo_path):
    """
        Global feature extraction used as this is shown to be working better
        (Wen & Guyer, 2012)
    """

    # load and convert photo to HSV
    photo = cv2.imread(photo_path)
    binary = binary_photo(photo)

    # label the foreground object (1) and exclude background (0)
    _, labels_binary = cv2.connectedComponents(binary)
    labels_binary = labels_binary.astype(np.uint8)

    # get features from photo for regions
    features = measure.regionprops(labels_binary)

    # Geometric features (Wen et al., 2009a,b)
    geometric_features_list = []
    for feature in features:
        if feature.label == 0:
            continue
        area = feature.area
        geometric_features_list.append(feature.area)

        perimeter = feature.perimeter
        geometric_features_list.append(feature.perimeter)

        circularity_ratio = (4*np.pi*area/((perimeter**2)+1))
        geometric_features_list.append(circularity_ratio)

        geometric_features_list.append(feature.eccentricity)
        geometric_features_list.append(feature.major_axis_length)

        geometric_features_list.append(feature.minor_axis_length)
        geometric_features_list.append(feature.convex_area)
        geometric_features_list.append(feature.solidity)
        geometric_features_list.append(feature.equivalent_diameter_area)

    return geometric_features_list


def fourier(photo_path):
    """
    Get spatial frequencies for contour detection
    (Bleed AI Academy, 2021; Wen & Guyer, 2012)
    Fourier transformation (OpenCV, n.d.)
    """

    # load photo
    photo = cv2.imread(photo_path)

    # get binary photo
    binary = binary_photo(photo)

    # detecting the contours in an photo
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # fourier transformation
    contour = max(contours, key=cv2.contourArea)

    fourier_contour = np.fft.fft2(contour)
    fourier_shift = np.fft.fftshift(fourier_contour)
    magnitude_spectrum = 20*np.log(np.abs(fourier_shift))

    magnitude_spectrum_first_two_spat_freq = magnitude_spectrum[:2]

    spat_freq_1 = magnitude_spectrum_first_two_spat_freq[0][0][0]
    spat_freq_2 = magnitude_spectrum_first_two_spat_freq[0][0][1]

    return spat_freq_1, spat_freq_2


def minimum_rectangle_photo(photo_path):
    """
    Get minimal rectangle photo (OpenCV, n.d.B)
    """

    photo = cv2.imread(photo_path)
    binary = binary_photo(photo)

    # get contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # get coordinates of minimum bounding rectangle
    if len(contours) > 0:
        contour = max(contours, key=cv2.contourArea)
        x_top_left, y_top_left, width, height = cv2.boundingRect(contour)
        cv2.rectangle(photo,
                      (x_top_left, y_top_left),
                      (x_top_left + width, y_top_left + height),
                      (0, 0, 255),
                      2)
        rectangle_photo = photo[y_top_left:y_top_left+height, x_top_left:x_top_left+width]

        if rectangle_photo.size > 0:
            pass
        else:
            print('Rectangle photo is empty')
    else:
        print('No contours found.')

    return rectangle_photo


def invariant_moments(photo_path):
    """
    Get invariant moments (Wen & Guyer, 2012; Tutorialspoint, n.d.)
    """
    rect_photo = minimum_rectangle_photo(photo_path)
    gray_rect_photo = cv2.cvtColor(rect_photo, cv2.COLOR_BGR2GRAY)

    moments = cv2.moments(gray_rect_photo)
    hu_moments = cv2.HuMoments(moments)

    return (hu_moments[0][0],
            hu_moments[1][0],
            hu_moments[2][0],
            hu_moments[3][0],
            hu_moments[4][0],
            hu_moments[5][0],
            hu_moments[6][0])


def texture(photo_path):
    """
    Take avarage of texture features (Wen & Guyer, 2012)
    """

    photo = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)

    # computer gray level co occurence
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    gray_level_co_occurence = graycomatrix(photo, distances, angles)

    # extract texture props of the GLCM
    contrast = graycoprops(gray_level_co_occurence, 'contrast')
    dissimilarity = graycoprops(gray_level_co_occurence, 'dissimilarity')
    homogeneity = graycoprops(gray_level_co_occurence, 'homogeneity')
    energy = graycoprops(gray_level_co_occurence, 'energy')
    correlation = graycoprops(gray_level_co_occurence, 'correlation')
    uniformity = graycoprops(gray_level_co_occurence, 'ASM')

    mean_contrast = contrast[0].mean(axis=0)
    mean_dissimilarity = dissimilarity[0].mean(axis=0)
    mean_homogeneity = homogeneity[0].mean(axis=0)
    mean_energy = energy[0].mean(axis=0)
    mean_correlation = correlation[0].mean(axis=0)
    mean_uniformity = uniformity[0].mean(axis=0)

    return (mean_contrast,
            mean_dissimilarity,
            mean_homogeneity,
            mean_energy,
            mean_correlation,
            mean_uniformity)


def color(photo_path):
    """
    Extract color features of minimum rectangle background object
    """
    def lab_to_lch(lab_photo):
        """ Convert LAB photos to LCH photos """
        luminence_lab, chrominance1_lab, chrominance2_lab = cv2.split(lab_photo)
        chrominance = np.sqrt(chrominance1_lab**2 + chrominance2_lab**2)
        hue = np.arctan2(chrominance2_lab, chrominance1_lab) * (180 / np.pi)
        hue[hue < 0] += 360

        luminence_lab = luminence_lab.astype(np.float32)
        chrominance = chrominance.astype(np.float32)
        hue = hue.astype(np.float32)

        return cv2.merge((luminence_lab, chrominance, hue))

    # Read the photo
    rect_photo = minimum_rectangle_photo(photo_path)
    photo = cv2.GaussianBlur(rect_photo, (5, 5), 0)

    # hsv photo features
    hsv_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2HSV)
    hue_hsv, saturation_hsv, _ = cv2.split(hsv_photo)

    mean_hue_hsv = np.mean(hue_hsv)
    std_hue_hsv = np.std(hue_hsv)
    mean_sat_hsv = np.mean(saturation_hsv)
    std_sat_hsv = np.std(saturation_hsv)

    # lab photo features
    lab_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB)
    luminence_lab, chrominance1_lab, chrominance2_lab = cv2.split(lab_photo)
    mean_luminance = np.mean(luminence_lab)
    saturation_lab = np.sqrt(np.square(chrominance1_lab) + np.square(chrominance2_lab))

    # scale saturation lab down to prevent overflow
    max_value = np.max(saturation_lab)
    if max_value > 0:
        saturation_lab_scaled = saturation_lab / max_value
    else:
        saturation_lab_scaled = saturation_lab

    # check if there is sufficient variability in the data
    if np.var(saturation_lab_scaled) > 0:
        std_sat_lab = np.std(saturation_lab_scaled)
    else:
        std_sat_lab = 0.0

    # lch photo features
    lch_photo = lab_to_lch(lab_photo)
    _, saturation_lch, hue_lch = cv2.split(lch_photo)

    mean_hue_lch = np.mean(hue_lch)
    std_hue_lch = np.std(hue_lch)
    mean_sat_lch = np.mean(saturation_lch)
    std_sat_lch = np.std(saturation_lch)

    return (mean_hue_hsv, std_hue_hsv, mean_sat_hsv,
            std_sat_hsv, mean_hue_lch, std_hue_lch,
            mean_sat_lch, std_sat_lch, mean_luminance,
            std_sat_lab)
