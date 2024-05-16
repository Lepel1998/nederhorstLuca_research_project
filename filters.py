# File for functions used in main file

import cv2
from PIL import Image, ImageFilter


# in this part of the code, the function will be used to enhance the image imported. Based on article by Makander & Halalli (2015), Gaussian filter works the best for both lowpass and highpass filters
## lowpass filter is to smooth the image to reduce noise and minor details

def LowpassFilter(photo):
    photo = Image.open(photo)
    photo = photo.filter(ImageFilter.GaussianBlur)
    return photo


# highpass filter is to extract edges and details from original photo
def HighpassFilter(photo):
    photo = Image.open(photo)
        

    return


# combine both results to enhance edges and details while keeping noise low


