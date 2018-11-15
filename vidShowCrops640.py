# import the necessary packages
import io
import time
import cv2
import numpy as np
import cropdata640
# import picamera
# from PIL import Image

mx=0
my=0
x=0
y=0
frameNo = 0

# crop_ranges are y,y1,x,x1 from top left
ball_crop_ranges = cropdata640.ballCrops
reset_crop_ranges = cropdata640.resetArmCrops
pin_crop_ranges = cropdata640.pin_crop_ranges

def drawPinRectangles():
    global pin_image
    global pin_crop_ranges
    global x
    global y   
    # NOTE: crop is img[y: y + h, x: x + w] 
    # cv2.rectangle is a = (x,y) , b=(x1,y1)
    for i in range(0,10):
        a =(pin_crop_ranges[i][2]+x,pin_crop_ranges[i][0]+y)
        b = (pin_crop_ranges[i][3]+x, pin_crop_ranges[i][1]+y)
        cv2.rectangle(pin_image, b, a, 255, 2)
        if i == 6:
            cv2.putText(pin_image,str(a),a,cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
            cv2.putText(pin_image,str(b),b,cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
    if frameNo==11:
        cv2.imwrite('C:/Users/cliff/OneDrive/pyProjects/videos/AAA/VidCombinedMask2.jpg',pin_image)
    else:
        print('frame11')
        cv2.imwrite('C:/Users/cliff/OneDrive/pyProjects/videos/AAA/VidPinMask2.jpg',pin_image)

def drawBallRectangles():
    global ball_image
    global ball_crop_ranges, reset_crop_ranges
    global mx
    global my
    # NOTE: crop is img[y: y + h, x: x + w] 
    # cv2.rectangle is a = (x,y) , b=(x1,y1)

    for i in range(0,1):
        a =(ball_crop_ranges[i][2]+mx,ball_crop_ranges[i][0]+my)
        b = (ball_crop_ranges[i][3]+mx, ball_crop_ranges[i][1]+my)
        cv2.rectangle(ball_image, b, a, 255, 2)
        c =(reset_crop_ranges[i][2]+mx,reset_crop_ranges[i][0]+my)
        d = (reset_crop_ranges[i][3]+mx, reset_crop_ranges[i][1]+my)
        cv2.rectangle(ball_image, d, c, 255, 2)
        if i == 0:
            cv2.putText(ball_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(ball_image,str(b),(b[0]-250,b[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imwrite('C:/Users/cliff/OneDrive/pyProjects/videos/AAA/VidBallMask2.jpg',ball_image)
    print('frame')

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
    # if frameNo == 11:
    #     ball_image = image
    #     drawBallRectangles()
    #     pin_image = ball_image
    #     drawPinRectangles()

    return True
       
# initialize the camera and grab a reference to the raw camera capture
cap = cv2.VideoCapture('../vidim/All/video0.h264')
while(cap.isOpened()):
    ret, frame2 = cap.read()
    if frameNo<15:
       detect_motion()
       
    else:
        print("Complete.  Images in videos/AAA")
        break
