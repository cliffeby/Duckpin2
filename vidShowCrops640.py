# import the necessary packages
import io
import time
import cv2
import numpy as np
# import picamera
# from PIL import Image

mx=0
my=0
x=0
y=0
frameNo = 0

# crop_ranges are y,y1,x,x1 from top left
mask_crop_ranges = ([300,475,20,580],[0,0,0,0])
crop_ranges = ([235,260, 315,340],[205,230, 290,315],[205,230,365,390],[180,205, 260,280],[180,205, 335,360],
    [180,205, 410,435],[155,180, 250,275],[155,180, 315,335],[155,180, 380,400],[155,180, 450,470])

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
            cv2.putText(pin_image,str(a),a,cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
            cv2.putText(pin_image,str(b),b,cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
    if frameNo==11:
        cv2.imwrite('../videos/AAA/AVidCombinedMask.jpg',pin_image)
    else:
        cv2.imwrite('../videos/AAA/AVidPinMask.jpg',pin_image)

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
    cv2.imwrite('../videos/AAA/AVidBallMask.jpg',ball_image)

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
cap = cv2.VideoCapture('../videos/AAA/video640.h264')
while(cap.isOpened()):
    ret, frame2 = cap.read()
    if frameNo<15:
       detect_motion()
    else:
        print("Complete.  Images in videos/AAA")
        break
