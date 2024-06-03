"""
Color feature extraction
Image used should be cropped image within minimal rectangle object
Did not work so far
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def lab_to_lch(lab_img):
    luminence_LAB, chrominance1_LAB, chrominance2_LAB = cv2.split(lab_img)
    C = np.sqrt(chrominance1_LAB**2 + chrominance2_LAB**2)
    H = np.arctan2(chrominance2_LAB, chrominance1_LAB) * (180 / np.pi)
    H[H<0] += 360

    luminence_LAB = luminence_LAB.astype(np.float32)
    C = C.astype(np.float32)
    H = H.astype(np.float32)
    return cv2.merge((luminence_LAB, C, H))

def display_images(images, titles, cmap=None):
    plt.figure(figsize=(15, 5))
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image, cmap=cmap if i > 0 else None)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

# Read the image
img = cv2.imread('image.png')
img = cv2.GaussianBlur(img, (5,5),0)

# hsv image features
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hue_HSV, saturation_HSV, value_HSV = cv2.split(hsv_image)

mean_hue_hsv = np.mean(hue_HSV)                         
std_hue_hsv = np.std(hue_HSV)                               
mean_sat_hsv = np.mean(saturation_HSV)  
std_sat_hsv = np.std(saturation_HSV)    

# Display the original image, HSV image, and the individual channels

# lab image features
lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
luminence_LAB, chrominance1_LAB, chrominance2_LAB = cv2.split(lab_image)

mean_luminance = np.mean(luminence_LAB) 

saturation_lab = np.sqrt(np.square(chrominance1_LAB) + np.square(chrominance2_LAB))
std_sat_lab = np.std(saturation_lab)


# lch image features

lch_image = lab_to_lch(lab_image)
lightness_LCH, saturation_LCH, hue_LCH = cv2.split(lch_image)

mean_hue_LCH = np.mean(hue_LCH)                
std_hue_LCH = np.std(hue_LCH)                  
mean_sat_LCH = np.mean(saturation_LCH)         
std_sat_LCH = np.std(saturation_LCH)           
print(mean_hue_LCH, std_hue_LCH, mean_sat_LCH, std_sat_LCH)

display_images(
    [img, lch_image, lightness_LCH, saturation_LCH, hue_LCH],
    ["Original Image (RGB)", "LAB Image", " luminence_LAB", "chrominance1_LAB", "chrominance2_LAB"],
    cmap='gray'
)



