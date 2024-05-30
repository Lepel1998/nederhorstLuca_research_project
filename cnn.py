"""
Module: Convolutional Neural Network using Tensorflow Keras API

This module contains the set-up of the model using Keras API.
"""

# 1. Setup and Load Data
## 1.1 Install dependencies and Setup
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import numpy as np

## 1.2 Remove dodgy images
#import cv2
#import imghdr

## 1.3 Load Data
data = "C:/Users/luca-/Documents/Forensic Science/year 2/Research/Research Project/AI/Processed_Dataset"
data = tf.keras.utils.image_dataset_from_directory(data, batch_size = 32, image_size = (67, 67))
data_iterator = data.as_numpy_iterator() # loop through data
batch = data_iterator.next()    # getting batch back
print(batch[0].shape)  # two pods --> images and labels






# see all photos inside folder












