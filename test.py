from functions import binary_photo
import cv2
import matplotlib.pyplot as plt


img = cv2.imread('S1#129 (1).jpg')
img = cv2.GaussianBlur(img, (5,5), 0)
img = cv2.GaussianBlur(img, (5,5), 0)

print(img)
binary = binary_photo(img)

plt.imshow(binary, cmap='gray')
plt.show()

contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 1:
    contour = max(contours, key=cv2.contourArea)
    #print(f'Number of contours found = {format(len(contour))}')
else:
    contour = contours
    #print(f'Number of contours found = {format(len(contour))}')

cv2.drawContours(img, contour, -1, (0,255,0),5)

plt.figure(figsize=[10,10])
plt.imshow(img[:,:,::-1])
plt.axis("off")
plt.show()
print(f'Number of contours found = {format(len(contour))}')
