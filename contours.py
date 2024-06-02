import cv2

photo_path = 'image.png'

photo = cv2.imread(photo_path)
grey_photo = cv2.cvtColor(photo,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(grey_photo, (5,5), 0)
canny = cv2.Canny(blur, threshold1 = 230, threshold2=250)

contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

contour_image = photo.copy()
cv2.drawContours(contour_image, contours, -1,(0,255,0), 2)

cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

