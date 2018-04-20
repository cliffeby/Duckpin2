import cv2
import numpy as np
import time
setterPresent = False
cap = cv2.VideoCapture('../videos/video2d.h264')
frameNo  =0
while(1):
    # Take each frame
    _, frame = cap.read()
    frameNo = frameNo+1
    if setterPresent:
        if firstSetterFrame + 34 > frameNo:
            continue    
    # Convert BGR to HSV
    frame = frame[150:450, 650:1600]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of green color in HSV
    lower_green = np.array([65,60,60])
    upper_green = np.array([80,255,255])
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask=mask)
    _,thrshed = cv2.threshold(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),3,255,cv2.THRESH_BINARY)
    _,contours,_ = cv2.findContours(thrshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    setterPresent = False
    area = 0
    for cnt in contours:
        #Contour area is taken
        area = cv2.contourArea(cnt) +area    
    if area >1000:
        setterPresent = True
        firstSetterFrame = frameNo
    if setterPresent:
        print("Green", area, frameNo)
    else:
        firstSetterFrame = 0
    
    cv2.imshow('frame',frame)

cap.release()
cv2.destroyAllWindows()
