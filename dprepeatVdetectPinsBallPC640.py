# import the necessary packages

import time
import cv2
import numpy
# import RPi.GPIO as GPIO
# from matplotlib import pyplot as plt

# crop_ranges are y,y1,x,x1 from top left
mask_crop_ranges = ([300,475,20,580],[0,0,0,0])
pin_crop_ranges = ([235,260, 315,340],[205,230, 290,315],[205,230,365,390],[180,205, 260,280],[180,205, 335,360],
    [180,205, 410,435],[155,180, 250,275],[155,180, 315,335],[155,180, 380,400],[155,180, 450,470])

def writeImageSeries(frameNoStart, numberOfFrames, img_rgb):
    if frameNoStart <= frameNo:
        if frameNo <= frameNoStart+numberOfFrames:
            print ('Saving ../videos/video3dFrame'+ str(frameNo) +'.jpg')
            cv2.imwrite('../videos/video3dFrame'+ str(frameNo) +'.jpg',img_rgb)

def isPinSetter():
    global setterPresent
    global frameNo
    global img_rgb
    global firstSetterFrame  
    # Convert BGR to HSV
    frame = img_rgb[150:450, 650:1600]
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
        global frame2
        pinCount = 0
        crop = []
        sumHist = [0,0,0,0,0,0,0,0,0,0]
        lower_red = numpy.array([0,0,100]) # lower_red = np.array([0,100,0])
        upper_red = numpy.array([110, 110, 255])  # upper_red = np.array([180,255,255])

        mask = cv2.inRange(img_rgb,lower_red,upper_red)
        output = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        threshold = 5
        for i in range(0,10):
                crop.append(output[pin_crop_ranges[i][0]+y:pin_crop_ranges[i][1]+y1,pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
                hist = cv2.calcHist([crop[i]],[1],None,[4], [10,50])
                sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
 
                if threshold < sumHist[i]:
                    pinCount = pinCount + 2**(9-i)
                
        print('HIST', frameNo, pinCount)
        # bit_GPIO(pinsGPIO,pinCount)

        if priorPinCount == pinCount:
            return False
        else:
            priorPinCount = pinCount
            return True
    
cap = cv2.VideoCapture('../videos/output2.mp4')
# setupGPIO(pinsGPIO)
setterPresent = False
armPresent = False
priorPinCount = 0
x=20
x1=0 +x
y=-0
y1=0 + y
crop_ranges = ([1200,1800, 220,2800],[0,0,0,0])

frameNo = 0
prevFrame = 0
ballCounter = [0]*3
origCounter = 0
for i in range(0,1):
    a =(int(crop_ranges[i][2]/2)+x,int(crop_ranges[i][0]/2)+y)
    b = (int(crop_ranges[i][3]/2)+x1, int(crop_ranges[i][1]/2)+y1)
ret, frame1 = cap.read()
mask= frame1[650:900, 250:1300]
frame1 = mask

while(cap.isOpened()):
    ret, frame2 = cap.read()
    try:
        type(frame2[0]) is None
    except:
        print ("New Video")
        cap.release()
        cap = cv2.VideoCapture('../videos/video3e.h264')
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
    frame2= frame2[650:900, 250:1300]
    img_gray1 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(img_gray1,img_gray2)
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
        print('radius', radius, frameNo, len(cnts))
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
            if center < (1100,200):
                    print('CENTER',center, radius, frameNo, len(cnts))
                    # cv2.imwrite('P:videos/cv2Img'+str(frameNo)+'.jpg',img_gray2)
            else:
                firstArmFrame = frameNo
                armPresent = True
    cv2.imshow('Ball', img_gray2)
    cv2.imshow('Thresh' , thresh)
    tf = findPins()

    cv2.rectangle(img_rgb,b, a, 255,2)

    # cv2.imshow('IMG_RGB with Ball Rect', img_rgb)
    # writeImageSeries(135,20)
    
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
