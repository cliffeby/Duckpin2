# import the necessary packages
import io
import time

import numpy as np

import cv2

# cap = cv2.VideoCapture('C:/Users/cliff/OneDrive/pyProjects/videos/video0e.h264')
# while(cap.isOpened()):
#     ret, frame2 = cap.read()
#     cv2.imshow('eee',frame2)

img = cv2.imread('C:/Users/cliff/pictures/BBallMaskh.jpg',1)
cv2.imshow('ddd',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

