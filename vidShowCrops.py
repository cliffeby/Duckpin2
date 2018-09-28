# import the necessary packages
import io
import time

import numpy as np

import cv2

# import picamera
# from PIL import Image

mx=0
my=0
x=20
y=0
frameNo = 0

# crop_ranges are y,y1,x,x1 from top left
mask_crop_ranges = ([500,850,100,1400],[0,0,0,0])
crop_ranges = ([390,490, 820,930],[325,425, 735,835],[345,445, 960,1060],[265,365, 680,760],[280,380, 860,960],
    [300,400, 1065,1165],[260,360, 600,680],[265,365, 790,890],[275,370, 970,1070],[280,380, 1150,1250])

def drawPinRectangles():
    global pin_image
    global crop_ranges
    global x
    global y   
    # NOTE: crop is img[y: y + h, x: x + w] 
    # cv2.rectangle is a = (x,y) , b=(x1,y1)
    for i in range(0,10):
        a =(crop_ranges[i][2]+x,crop_ranges[i][0]+y)
        b = (crop_ranges[i][3]+x, crop_ranges[i][1]+y)
        cv2.rectangle(pin_image, b, a, 255, 2)
        if i == 6:
            cv2.putText(pin_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(pin_image,str(b),b,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    if frameNo==11:
        cv2.imwrite('C:/Users/cliff/OneDrive/pyProjects/videos/AAA1/AVidCombinedMask.jpg',pin_image)
        print ('C:/Users/cliff/OneDrive/pyProjects/videos/AAA1/AVidCombinedMask.jpg')
    else:
        cv2.imwrite('C:/Users/cliff/OneDrive/pyProjects/videos/AAA1/AVidPinMask.jpg',pin_image)

def drawBallRectangles():
    global ball_image
    global mask_crop_ranges
    global mx
    global my
    # NOTE: crop is img[y: y + h, x: x + w] 
    # cv2.rectangle is a = (x,y) , b=(x1,y1)

    for i in range(0,1):
        a =(mask_crop_ranges[i][2]+mx,mask_crop_ranges[i][0]+my)
        b = (mask_crop_ranges[i][3]+mx, mask_crop_ranges[i][1]+my)
        cv2.rectangle(ball_image, b, a, 255, 2)
        if i == 0:
            cv2.putText(ball_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(ball_image,str(b),(b[0]-250,b[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imwrite('C:/Users/cliff/OneDrive/pyProjects/videos/AAA/AVidBallMask.jpg',ball_image)

def detect_motion():
    global frameNo
    global mask
    global pin_image
    global ball_image
    global frame2
    
    frameNo = frameNo +1
    image = frame2
    
    
    if frameNo == 9:
        pin_image = image
        drawPinRectangles()
    if frameNo == 10:
        ball_image = image
        drawBallRectangles()
    if frameNo == 11:
        ball_image = image
        drawBallRectangles()
        pin_image = ball_image
        drawPinRectangles()

    return True
       
# initialize the camera and grab a reference to the raw camera capture
cap = cv2.VideoCapture('C:/Users/cliff/OneDrive/pyProjects/videos/dp _1023_277_.h264')
while(cap.isOpened()):
    ret, frame2 = cap.read()
    cv2.imshow('eee',frame2)
    print (frameNo)
    if frameNo<1500:
       detect_motion()
    else:
        print("Complete.  Images in videos/AAA")
        break
