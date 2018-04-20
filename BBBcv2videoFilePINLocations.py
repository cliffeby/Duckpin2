# import the necessary packages
import io
import time
import cv2
import numpy as np
import picamera
import RPi.GPIO as GPIO
from matplotlib import pyplot as plt
from matplotlib import style
from picamera import Color, PiCamera
from picamera.array import PiRGBArray
from PIL import Image

global img_rgb
global current_image
global resFactor
global counter
global img_gray_temp
global crop_ranges
global priorPinCount

pinsGPIO = [6, 26, 20, 5, 21, 3, 16, 2, 14, 15]
lower_red = np.array([0,0,55]) # lower_red = np.array([0,100,0])
upper_red = np.array([110, 110, 255])  # upper_red = np.array([180,255,255])
prior_image = None
img_gray1 = None
PBFN = 0
PBFD= []
newSeries = True
mx=0
mx1=0 +mx
my=0
my1=0 + my
mask_crop_ranges = ([1100,1700, 220,2800],[0,0,0,0])
crop_ranges = ([390,490, 820,930],[325,425, 735,835],[345,445, 960,1060],[265,365, 680,760],[280,380, 860,960],
    [300,400, 1065,1165],[260,360, 600,680],[265,365, 790,890],[275,370, 970,1070],[280,380, 1150,1250])
resFactor = 1
pts = []
kernel = np.ones((5,5), np.uint8)
frameNo = 0
prevFrame = 0
ballCounter = [0]*3
origCounter = 0
priorPinCount = 0
for i in range(0,1):
    a =(int(mask_crop_ranges[i][2]/2)+mx,int(mask_crop_ranges[i][0]/2)+my)
    b = (int(mask_crop_ranges[i][3]/2)+mx1, int(mask_crop_ranges[i][1]/2)+my1)
# ret, frame1 = cap.read()
# frame1= frame1[1100:220, 1700:2850]
# mask= frame1[550:850, 250:1600]
# frame1 = mask

def isFrameNext(current, previous):
    if current-previous == 1:
        return True
    else:
        return False
 
def ballOrReset( frame_img,center):
    global PBFN, PBFD,mask
    if PBFN > 6:
        return
    if PBFN == 0:
        PBFD = []
        mask = frame_img
    if PBFN < 7:
        PBFD.append(center) #append data
        PBFN = PBFN+1
        return

def setupGPIO(pins):
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in pins:
        GPIO.setup(pin,GPIO.OUT)
        print ("setup Completed")

def bit_GPIO(pins,pinCount):
    bits = "{0:b}".format(pinCount)
    while len(bits)<10:
        bits = "0"+bits
    for idx in range(0,len(bits)):
        # print(idx,bits, bits[idx])
        if(bits[idx]=="1"):
             GPIO.output(pins[idx], GPIO.HIGH)
        else:
            GPIO.output(pins[idx], GPIO.LOW)

def drawPinRectangles():
    global current_image
    global resFactor
    x=-90
    x1=+x
    y=-150
    y1=+ y
    # crop_img = img_rgb[250:600, 600:1300] #all ten
    # NOTE: its img[y: y + h, x: x + w] 
    for i in range(0,10):
        a =(crop_ranges[i][2]+x,crop_ranges[i][0]+y)
        b = (crop_ranges[i][3]+x1, crop_ranges[i][1]+y1)
        a1 = (int(a[0]/resFactor), int(a[1]/resFactor))
        b1 = (int(b[0]/resFactor), int(b[1]/resFactor))
        # print( a,b,a1, b1)
        cv2.rectangle(current_image, b1, a1, 255, 2)
    # cv2.imshow('BO.jpg', current_image) 

def detect_motion(camera):
    global prior_image
    global priorPinCount
    global current_image
    global counter
    global frameNo
    global mask
    global img_gray1
    global prevFrame
    global PBFN
    global newSeries
    x=-90
    x1= +x
    y=-150
    y1=+ y
    stream = io.BytesIO()
    camera.rotation = 270
    camera.capture(stream, format='jpeg', use_video_port=True)
    stream.seek(0)
    if prior_image is None:
        prior_image = Image.open(stream)
        prior_image = np.fromstring(stream.getvalue(), dtype=np.uint8)
        prior_image = cv2.imdecode(prior_image,1)
        mask = prior_image[550:850, 250:1600]
        img_gray1 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        return False
    else:
        current_image = np.fromstring(stream.getvalue(), dtype=np.uint8)
        # current_image = np.array(current_image)
        current_image = cv2.imdecode(current_image,1)

        frame2 = current_image
        frameNo = frameNo +1
        img_rgb = frame2
        frame2= frame2[550:850, 250:1600]
        
        img_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(img_gray1,img_gray2)
        ret, thresh = cv2.threshold(diff, 100,200,cv2.THRESH_BINARY)
        frame = thresh
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
    
        # only proceed if at least one contour was found
        # if len(cnts) > 3:
        #     continue
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            ballCounter[0]=0
            ballCounter[1]=0
            ballCounter[2]=0
            # only proceed if the radius meets a minimum size
            if radius > 20:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                if center < (1100,200):
                    if newSeries==True:
                        prevFrame = frameNo-1
                        newSeries = False
                        PBFN = 0
                    if isFrameNext(frameNo, prevFrame):
                        prevFrame = frameNo
                        ballOrReset(mask,center)

                        print('CENTER',center, radius, frameNo, len(cnts), PBFN)
                    else:
                        print('Final',frameNo, PBFD)
                        newSeries = True

            # # show the frame to our screen
            # cv2.imshow("Frame", frame)
        
        # cv2.rectangle(img_rgb,b, a, 255,2)
        # cv2.imshow("Frame1/Mask", mask)
        # cv2.imshow('BO.jpg', img_gray1)
        # cv2.imshow('Diff', thresh)
        # xx=137
        # if frameNo==xx:
        #     cv2.imwrite('../videos/videodFrame'+ str(xx) +'.jpg',img_rgb)
        # if frameNo==xx+1:
        #     cv2.imwrite('../videos/videodFrame'+ str(xx+1) +'.jpg',img_rgb)
        # if frameNo==xx+2:
        #     cv2.imwrite('../videos/videodFrame'+ str(xx+2) +'.jpg',img_rgb)
        # if frameNo==xx+3:
        #     cv2.imwrite('../videos/videodFrame'+ str(xx+3) +'.jpg',img_rgb)
        # if frameNo==xx+4:
        #     cv2.imwrite('../videos/videodFrame'+ str(xx+4) +'.jpg',img_rgb)
        # if frameNo==xx+5:
        #     cv2.imwrite('../videos/videodFrame'+ str(xx+5) +'.jpg',img_rgb)
        # if frameNo==xx+6:
        #     cv2.imwrite('../videos/videodFrame'+ str(xx+6) +'.jpg',img_rgb)


        drawPinRectangles()
        cv2.imwrite('/home/pi/Shared/videos/bImg'+str(frameNo)+'.jpg',current_image)

        img_rgb = current_image

        mask = cv2.inRange(img_rgb,lower_red,upper_red)
        mask = cv2.erode(mask, None, iterations=1)
        mask = cv2.dilate(mask, None, iterations=1)
        output = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        # cv2.imshow('Mask', mask)
        # cv2.imwrite('/home/pi/Shared/videos/bImgMask'+str(counter)+'.jpg', output)
        # cv2.imshow('Frame',current_image)
        # cv2.imshow('Output',output)
        sumHist = [0,0,0,0,0,0,0,0,0,0]

        pinCount = 0
        crop = []
        sumHist = [0,0,0,0,0,0,0,0,0,0]
        x=-90
        x1= x
        y=-150
        y1= y
        for i in range(0,10):
                # print(crop_ranges[i][0]+y,crop_ranges[i][1]+y1,crop_ranges[i][2]+x,crop_ranges[i][3]+x1)
                # cv2.imwrite('/home/pi/Shared/videos/bImgMaskPin'+str(i*frameNo)+'.jpg', img_rgb[crop_ranges[i][0]+y:crop_ranges[i][1]+y1,crop_ranges[i][2]+x:crop_ranges[i][3]+x1])
                crop.append(output[crop_ranges[i][0]+y:crop_ranges[i][1]+y1,crop_ranges[i][2]+x:crop_ranges[i][3]+x1])
                hist = cv2.calcHist([crop[i]],[1],None,[4], [10,50])
                sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
                
                threshold = 25
            
                if threshold < sumHist[i]:
                    pinCount = pinCount + 2**(9-i)
                
        print('HIST', frameNo, sumHist, pinCount)
        
                        
            #             plt.scatter(top_left[1], top_left[0])
            #             plt.text(top_left[1], top_left[0], str(i+1)+' '+str(top_left[1])+' '+str(top_left[0])+' '+str(min_val))
        bit_GPIO(pinsGPIO,pinCount)
        # plt.show()

        # counter = counter +1
        if priorPinCount == pinCount:
            return False
        # prior_image = current_image
        else:
            priorPinCount = pinCount
            return True
    
# initialize the camera and grab a reference to the raw camera capture



with picamera.PiCamera() as camera:
    setupGPIO(pinsGPIO)
    camera.resolution = (1440, 900)
    # camera.brightness = 45
    # camera.rotation = 270
    stream = picamera.PiCameraCircularIO(camera, seconds=5)
    camera.start_recording(stream, format='h264')
    try:
        while True:
            camera.wait_recording(1)
            if detect_motion(camera):
                print('Motion detected!')
                # As soon as we detect motion, split the recording to
                # record the frames "after" motion
                camera.split_recording('after.h264')
                # Write the 10 seconds "before" motion to disk as well
                stream.copy_to('/home/pi/Shared/videos/bffl'+str(frameNo)+'.h264', seconds=5)
                stream.clear()
                # Wait until motion is no longer detected, then split
                # recording back to the in-memory circular buffer
                while detect_motion(camera):
                    camera.wait_recording(1)
                print('Motion stopped!')
                camera.split_recording(stream)
    finally:
        camera.stop_recording()

