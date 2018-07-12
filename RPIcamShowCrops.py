# import the necessary packages
import io
import sys
import time
import cv2
import numpy as np
import picamera
from PIL import Image

mx=0
my=0
x=-11
y=0
frameNo = 0

def setResolution():
    resX = 1440
    resY = 900
    res = (int(resX), int(resY))
    return res

# crop_ranges are y,y1,x,x1 from top left
mask_crop_ranges = ([400,897,10,1096],[0,0,0,0])
crop_ranges = ([272,298,549,575],[218,244,493,519],[218,244,677,703],[177,203,451,477],[167,193,612,638],[168,194,776,802],
    [123,149,418,444],[125,151,564,590],[126,152,709,735],[124,150,867,893])
arm_crop_ranges = ([71,279,1025,1120],[0,0,0,0])
def drawPinRectangles():
    global pin_image
    global crop_ranges
    global arm_crop_ranges
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
        cv2.imwrite('/home/pi/Shared/videos/AAA/BCombinedMask.jpg',pin_image)
    else:
        cv2.imwrite('/home/pi/Shared/videos/AAA/BPinMask.jpg',pin_image)

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
    cv2.imwrite('/home/pi/Shared/videos/AAA/BBallMask.jpg',ball_image)

def drawArmRectangles():
    global arm_image
    global arm_crop_ranges
    global mx
    global my
    # NOTE: crop is img[y: y + h, x: x + w] 
    # cv2.rectangle is a = (x,y) , b=(x1,y1)

    for i in range(0,1):
        a =(arm_crop_ranges[i][2]+mx,arm_crop_ranges[i][0]+my)
        b = (arm_crop_ranges[i][3]+mx, arm_crop_ranges[i][1]+my)
        cv2.rectangle(ball_image, b, a, 255, 2)
        if i == 0:
            cv2.putText(ball_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(ball_image,str(b),(b[0]-250,b[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imwrite('/home/pi/Shared/videos/AAA/BArmMask.jpg',arm_image)

def detect_motion(camera):
    global frameNo
    global mask
    global pin_image
    global ball_image
    global arm_image
    
    stream = io.BytesIO()
    camera.capture(stream, format='jpeg', use_video_port=True)
    stream.seek(0)
    frameNo = frameNo +1
    Image.open(stream)
    image = np.fromstring(stream.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(image,1)
    
    if frameNo == 9:
        ball_image = image
        drawBallRectangles()
        arm_image = image
        drawArmRectangles()
    if frameNo == 10:
        pin_image = image
        drawPinRectangles()
    if frameNo == 11:
        ball_image = image
        drawBallRectangles()
        drawArmRectangles()
        pin_image = ball_image
        drawPinRectangles()
    return True
       
# initialize the camera and grab a reference to the raw camera capture

with picamera.PiCamera() as camera:
    camera.rotation = 180
    camera.resolution = setResolution()
    stream = picamera.PiCameraCircularIO(camera, seconds=5)
    camera.start_recording(stream, format='h264')
    try:
        while frameNo<15:
            # camera.wait_recording(1)
            if detect_motion(camera):
                # print('Motion detected!')
                # # As soon as we detect motion, split the recording to
                # # record the frames "after" motion
                # camera.split_recording('after.h264')
                # # Write the 10 seconds "before" motion to disk as well
                # stream.copy_to('/home/pi/Shared/videos/bbffl'+str(frameNo)+'.h264', seconds=5)
                # stream.clear()
                # # Wait until motion is no longer detected, then split
                # # recording back to the in-memory circular buffer
                # while detect_motion(camera):
                #     camera.wait_recording(1)
                # print('Motion stopped!')
                camera.split_recording(stream)
    finally:
        camera.stop_recording()
        print("Complete.  Images in videos/AAA")

