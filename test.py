import numpy as np
import cv2
from functions import binary_image
import matplotlib.pyplot as plt

def minimum_rectangle_image(img):
    """
        https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    """

    img = cv2.imread('vlieg.jpeg')
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
            plt.imshow(rectangle_image)
            plt.title('Extracted Bouding Rectangle')
            plt.show()
        else:
            print('Rectangle image is empty')
    else:
        print('No contours found.')

    return rectangle_image

minimum_rectangle_image('vlieg.jpeg')



def invariant_moments(photo_path):
    """
    https://www.tutorialspoint.com/how-to-compute-hu-moments-of-an-image-in-opencv-python
    """
    rect_image = minimum_rectangle_image(photo_path)
    gray_rect_image = cv2.cvtColor(rect_image, cv2.COLOR_BGR2GRAY)

    moments = cv2.moments(gray_rect_image)
    hu_moments = cv2.HuMoments(moments)

    return hu_moments[0][0], hu_moments[1][0], hu_moments[2][0], hu_moments[3][0], hu_moments[4][0], hu_moments[5][0], hu_moments[6][0]



invariant_moments('vlieg.jpeg')
