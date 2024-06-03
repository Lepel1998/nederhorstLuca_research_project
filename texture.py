import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# load image
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# computer gray level co occurence
distances=[1]
angles=[0, np.pi/4,np.pi/2, 3*np.pi/4]
gray_level_co_occurence = graycomatrix(img, distances,angles)

# extract texture props of the GLCM
contrast = graycoprops(gray_level_co_occurence, 'contrast')
dissimilarity = graycoprops(gray_level_co_occurence, 'dissimilarity')
homogeneity = graycoprops(gray_level_co_occurence, 'homogeneity')
energy = graycoprops(gray_level_co_occurence, 'energy')
correlation = graycoprops(gray_level_co_occurence,'correlation')
ASM = graycoprops(gray_level_co_occurence, 'ASM')

