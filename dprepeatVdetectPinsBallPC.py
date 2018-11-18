# import the necessary packages

import time
import cv2
import numpy
import cropdata640
import statistics, collections
# import RPi.GPIO as GPIO
# from matplotlib import pyplot as plt

pinsGPIO = [6, 26, 20, 5, 21, 3, 16, 2, 14, 15]
mask_crop_ranges = cropdata640.ballCrops
pin_crop_ranges = cropdata640.pin_crop_ranges
resetArmCrops = cropdata640.resetArmCrops
pinSetterCrops = cropdata640.pinSetterCrops

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
        if(bits[idx]=="1"):
             GPIO.output(pins[idx], GPIO.HIGH)
        else:
            GPIO.output(pins[idx], GPIO.LOW)

def getCroppedImage(image,crop_array):
    croppedImage = image[crop_array[0]:crop_array[1],crop_array[2]:crop_array[3]]
    return croppedImage


def writeImageSeries(frameNoStart, numberOfFrames, img_rgb):
    if frameNoStart <= frameNo:
        if frameNo <= frameNoStart+numberOfFrames:
            print ('Saving ../videos/video3dFrame'+ str(frameNo) +'.jpg')
            cv2.imwrite('../videos/video3dFrame'+ str(frameNo) +'.jpg',img_rgb)
            drawPinRectangles()

def isPinSetter():
    global setterPresent
    global frameNo
    global img_rgb
    global firstSetterFrame  
    # Convert BGR to HSV
    frame = img_rgb[50:150, 200:400]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of green color in HSV
    lower_green = numpy.array([65,60,60])
    upper_green = numpy.array([80,255,255])
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(frame,frame, mask=mask)
    _,thrshed = cv2.threshold(cv2.cvtColor(res,cv2.COLOR_BGR2GRAY),3,255,cv2.THRESH_BINARY)
    _,contours,_ = cv2.findContours(thrshed,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    setterPresent = False
    area = 0
    for cnt in contours:
        #Contour area is measured
        area = cv2.contourArea(cnt) +area
    if area >10000:
        setterPresent = True
        firstSetterFrame = frameNo
    if setterPresent:
        print("Green", area, frameNo)
    else:
        firstSetterFrame = 0

def arm():
    global firstArmFrame
    global frameNo
    firstArmFrame = frameNo

def findPins():
        global x,x1,y,y1
        global priorPinCount
        global img_rgb
        global frame2, frameNo
        pinCount = 0
        crop = []
        sumHist = [0,0,0,0,0,0,0,0,0,0]
        lower_red = numpy.array([0,180,80])
        upper_red = numpy.array([10,220,190])       
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv,lower_red,upper_red)
        output = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
        threshold = 1
        for i in range(0,10):
                crop.append(output[pin_crop_ranges[i][0]+y:pin_crop_ranges[i][1]+y1,pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
                hist = cv2.calcHist([crop[i]],[1],None,[4], [1,255])
                sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
                # if frameNo==2376:
                #     cv2.imwrite('../videos/pin'+ str(frameNo) +str(i+1)+'.jpg',crop[i])
                if threshold <= sumHist[i]:
                    pinCount = pinCount + myModeFilter(i)
        if frameNo%20 == 0:        
            print('HIST', frameNo, pinCount)
        # bit_GPIO(pinsGPIO,pinCount)

        if priorPinCount == pinCount:
            return False
        else:
            priorPinCount = pinCount
            # print(frameNo, pinCount, sumHist[6], sumHist[7], sumHist[8], sumHist[9])
            return True


def myModeFilter(index):
    global pinCounts
    pinCounts[index].append(1)
    newValue = statistics.mode(pinCounts[index])
    if newValue ==1:
        return 2**(9-index)
    else:
        return 0

def drawPinRectangles():
    global ball_image,img_rgb,x,y
    global pin_crop_ranges, mask_crop_ranges
    mx=x
    my=y
    ball_image = img_rgb
    ballCrops = mask_crop_ranges
    # NOTE: crop is img[y: y + h, x: x + w] 
    # cv2.rectangle is a = (x,y) , b=(x1,y1)

    for i in range(0,9):
        a =(pin_crop_ranges[i][2]+mx,pin_crop_ranges[i][0]+my)
        b = (pin_crop_ranges[i][3]+mx, pin_crop_ranges[i][1]+my)
        cv2.rectangle(ball_image, b, a, 255, 2)
        if i == 6:
            cv2.putText(ball_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(ball_image,str(b),(b[0]-250,b[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imwrite('C:/Users/cliff/OneDrive/pyProjects/videos/CCEPinMask'+str(i) +'.jpg',ball_image)
    a = (ballCrops[2]+mx,ballCrops[0]+my)
    b = (ballCrops[3]+mx, ballCrops[1]+my)
    cv2.rectangle(ball_image, b, a, 255, 2)
    cv2.putText(ball_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(ball_image,str(b),(b[0]-250,b[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imwrite('C:/Users/cliff/OneDrive/pyProjects/videos/CCEBBallMask'+str(i) +'.jpg',ball_image)
    a = (resetArmCrops[2]+mx, resetArmCrops[0]+my)
    b = (resetArmCrops[3]+mx, resetArmCrops[1]+my)
    cv2.rectangle(ball_image, b, a, 255, 2)
    cv2.putText(ball_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(ball_image,str(b),(b[0]-250,b[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imwrite('C:/Users/cliff/OneDrive/pyProjects/videos/CCEBBallMask'+str(i) +'.jpg',ball_image)
    
cap = cv2.VideoCapture('C:/Users/cliff/OneDrive/pyProjects/videos/640/video0.h264')
# setupGPIO(pinsGPIO)
setterPresent = False
armPresent = False
priorPinCount = 0
bitBuckets =  collections.deque(9*[1], 9)
pinCounts =[bitBuckets for x in range(10)]
x=0
x1=0 +x
y=-0
y1=0 + y
crop_ranges = mask_crop_ranges

frameNo = 0
prevFrame = 0
ballCounter = [0]*3
origCounter = 0
# for i in range(0,1):
#     a =(int(crop_ranges[i][2]/2)+x,int(crop_ranges[i][0]/2)+y)
#     b = (int(crop_ranges[i][3]/2)+x1, int(crop_ranges[i][1]/2)+y1)
ret, frame1 = cap.read()
mask= getCroppedImage(frame1,mask_crop_ranges)
frame1 = mask

while(cap.isOpened()):
    
    ret, frame2 = cap.read()
    try:
        type(frame2[0]) is None
    except:
        print ("New Video")
        cap.release()
        cap = cv2.VideoCapture('C:/Users/cliff/OneDrive/pyProjects/videos/640/video0.h264')
        ret, frame2 = cap.read()
    frameNo = frameNo +1
    img_rgb = frame2

    if setterPresent:
            if firstSetterFrame + 120 > frameNo:
                continue
    if armPresent:
            if firstArmFrame + 120 > frameNo:
                continue
            if firstArmFrame+120 == frameNo:
                armPresent = False

    isPinSetter()
    frame2= getCroppedImage(frame2, mask_crop_ranges)
    
    img_gray1 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(img_gray1,img_gray2)
    diff2 = cv2.subtract(img_gray1,img_gray2)
    diff_count = cv2.countNonZero(diff2)
    # if diff2 is True:
    #     print('SAME')
    # else:
    #     cv2.imshow('DIFF2', diff2)
    #     print("different", diff_count)
    # First value reduces noise.  Values above 150 seem to miss certain ball colors
    ret, thresh = cv2.threshold(diff, 120,255,cv2.THRESH_BINARY)
    frame = thresh
    # Blur eliminates noise by averaging surrounding pixels.  Value is array size of blur and MUST BE ODD
    thresh = cv2.medianBlur(thresh,9)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    radius = 0
    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((xContour, yContour), radius) = cv2.minEnclosingCircle(c)
        # print('radius', radius, frameNo, len(cnts))
        ballCounter[0]=0
        ballCounter[1]=0
        ballCounter[2]=0
		# only proceed if the radius meets a minimum size
        if radius > 20:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.drawContours(img_gray2, cnts, -1, (0,255,0), 3)
            if center < (510,480):
                    print('CENTER',center, radius, frameNo, len(cnts))
                    # cv2.imwrite('P:videos/cv2Img'+str(frameNo)+'.jpg',img_gray2)
                    continue
            else:
                firstArmFrame = frameNo
                armPresent = True
    cv2.imshow('All', img_rgb)
    cv2.imshow('Ball', img_gray2)
    cv2.imshow('Thresh' , thresh)
    tf = findPins()

    # cv2.rectangle(img_rgb,b, a, 255,2)
    frame1 = frame2

    # cv2.imshow('IMG_RGB with Ball Rect', img_rgb)
    writeImageSeries(100,1,img_rgb)
    
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
