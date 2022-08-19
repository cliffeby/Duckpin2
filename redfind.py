import numpy as np
import cv2

image = cv2.imread('/home/cliffeby/Downloads/imgdpThursday (1).jpg')
result = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# Set using redslide on a 08/19/22 image
lower = np.array([0,42,0])
upper = np.array([16,255,126])
mask = cv2.inRange(image, lower, upper)
result = cv2.bitwise_and(result, result, mask=mask)

cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey()