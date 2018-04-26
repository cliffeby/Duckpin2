# import the necessary packages
import io
import sys
import time
import cv2
import numpy as np
import picamera
from PIL import Image
print(sys.argv)
if len(sys.argv) != 2:
    print('Add parameter "HIGH" or "LOW"')
mx=0
my=0
x=-50
y=-165
frameNo = 0

# crop_ranges are y,y1,x,x1 from top left
mask_crop_ranges = ([500,900,100,1300],[0,0,0,0])
crop_ranges = ([445,515, 755,825],[360,440, 710,755],[370,450, 885,940],[320,385, 655,705],[320,385, 815,875],
    [330,380, 980,1035],[275,345, 605,665],[275,345, 745,805],[275,345, 895,955],[275,345, 1060,1120])
if sys.argv[1]=="LOW":
    height = 912/480
    width = 1440/640
    x = int(x/width)
    y = int(y/height)
    mask_crop_ranges[0][0] = int(mask_crop_ranges[0][0]/height)
    mask_crop_ranges[0][1] = int(mask_crop_ranges[0][1]/height)
    mask_crop_ranges[0][2] = int(mask_crop_ranges[0][2]/width)
    mask_crop_ranges[0][3] = int(mask_crop_ranges[0][3]/width)
    for ii in range(0,10):
        print('CRB', crop_ranges[ii])
        crop_ranges[ii][0] = int(crop_ranges[ii][0]/height)
        crop_ranges[ii][1] = int(crop_ranges[ii][1]/height)
        crop_ranges[ii][2] = int(crop_ranges[ii][2]/width)
        crop_ranges[ii][3] = int(crop_ranges[ii][3]/width)
        print('CRA', crop_ranges[ii])

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
        cv2.imwrite('/home/pi/Shared/videos/AAA/ACombinedMask.jpg',pin_image)
    else:
        cv2.imwrite('/home/pi/Shared/videos/AAA/APinMask.jpg',pin_image)

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
    cv2.imwrite('/home/pi/Shared/videos/AAA/ABallMask.jpg',ball_image)

def detect_motion(camera):
    global frameNo
    global mask
    global pin_image
    global ball_image
    
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
    if frameNo == 10:
        pin_image = image
        drawPinRectangles()
    if frameNo == 11:
        ball_image = image
        drawBallRectangles()
        pin_image = ball_image
        drawPinRectangles()

    return True
       
# initialize the camera and grab a reference to the raw camera capture

with picamera.PiCamera() as camera:
    if sys.argv[1] == 'HIGH':
        camera.resolution = (1440, 900)
    if sys.argv[1] == "LOW":
        camera.resolution = (640,480)
    # camera.brightness = 45
    print("SYSARRGV", sys.argv[1])
    camera.rotation = 180
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

