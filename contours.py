"""
https://www.youtube.com/watch?v=JfaZNiEbreE&list=PLCeWwpzjQu9gc9C9-iZ9WTFNGhIq4-L1X


"""


import cv2
import os
import matplotlib.pyplot as plt
import numpy as np


# load image
img = cv2.imread('vlieg.jpeg')
img_copy = img.copy()

# preprocessing of image for contours using thresholding
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.GaussianBlur(gray_img, (5,5), 0)

# you need to invert image 
gray_inverted = cv2.bitwise_not(gray_img)

# make image fully black and white with cv2.threshold()
_,binary = cv2.threshold(gray_inverted, 40, 255, cv2.THRESH_BINARY)
plt.imshow(binary, cmap='gray')

# detecting the contours in an image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f'Number of contours found = {format(len(contours))}')

# read coloured image in for drawing the contours (otherwise the lines will be drawed in white)
cv2.drawContours(img, contours, -1, (0,255,0),1)
plt.figure(figsize=[10,10])
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()





"""
# load image
img = cv2.imread('vlieg.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.GaussianBlur(img, (5,5), 0)

# edge detection
canny = cv2.Canny(img, threshold1=75, threshold2=210)

# find contours (assuming insect is largest)
contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
insect_contour = max(contours, key=cv2.contourArea)

# calculate Fourier descriptor
contour_complex = insect_contour[:,0,0] + 1j * insect_contour[:,0,1]
fourier = np.fft.fft(contour_complex)

# extract two power spectra (first two spatial frequencies)
sorted_indices = np.argsort(np.abs(fourier))[::-1]
first_frequency= np.abs(fourier[sorted_indices[0]])
second_frequency = np.abs(fourier[sorted_indices[1]])
print(f'first frequency: {first_frequency}, second frequency: {second_frequency}')

# find moments of contours
cnt = contours[0]
M = cv2.moments(cnt)

# find hu moments using HuMoments(M) function for particular contour
Hm = cv2.HuMoments(M)

# draw contours on input image
cv2.drawContours(img, [cnt], -1, (0,255,255), 3)

# print
print("Hu-Moments of first contour:\n", Hm)
cv2.imshow("Hu-Moments", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

