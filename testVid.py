# import the necessary packages
import io
import time
import cropdata1024, cropdata1440
import numpy as np
import threading
import cv2

mask_crop_ranges = cropdata1440.ballCrops
crop_ranges = cropdata1024.pin_crop_ranges
arm_crop_ranges = cropdata1440.resetArmCrops
scrop_ranges = cropdata1024.special_crop_ranges
x=y=x1=y1=0
rmax = [0,0,0,0,0,0,0,0,0,0,0,-1]
smax = [0,0,0]
oldHist =olb=olg=olr=oub=oug=our = -999

def drawPinRectangles(pin_image):
    global crop_ranges,scrop_ranges
    global arm_crop_ranges
    global x,y,x1,y1  
    # NOTE: crop is img[y: y + h, x: x + w] 
    # cv2.rectangle is a = (x,y) , b=(x1,y1)
    for i in range(0,10):
        a =(crop_ranges[i][2]+x,crop_ranges[i][0]+y)
        b = (crop_ranges[i][3]+x1, crop_ranges[i][1]+y1)
        cv2.rectangle(pin_image, b, a, 255, 2)
        # if i == 6:
        #     cv2.putText(pin_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        #     cv2.putText(pin_image,str(b),b,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    for i in range(0,3):
        a =(scrop_ranges[i][2]+x,scrop_ranges[i][0]+y)
        b = (scrop_ranges[i][3]+x1, scrop_ranges[i][1]+y1)
        cv2.rectangle(pin_image, b, a, 255, 2)
    
def isGreater(num,i):
    global rmax
    if num>rmax[i]:
        rmax[i] = num
        return True
    else:
        return False

def isGreaterSpecial(num,i):
    global smax
    if num>smax[i]:
        smax[i] = num
        return True
    else:
        return False

def findPins(img_rgb):        
        global priorPinCount, frameNo
        global crop_ranges, scrop_ranges,pin_crop_ranges, sumHist
        global x,y,x1,y1,oldHist, olb,olg,olr,oub,oug,our,crop
        pin_crop_ranges = crop_ranges
        pinCount = 0
        crop = []
        scrop = []
        sumHist = [0,0,0,0,0,0,0,0,0,0]
        lower_red = np.array([0,0,50]) # lower_red = np.array([0,100,0])  try 0,0,50
        upper_red = np.array([150, 150, 240])  # upper_red = np.array([180,255,255])

        mask = cv2.inRange(img_rgb,lower_red,upper_red)
        output = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        # for y in range (10,0):
        #     # NOTE: crop is img[y: y + h, x: x + w] 
        #     # cv2.rectangle is a = (x,y) , b=(x1,y1)
        #     y1=-y
        #     x1=-y
        #     x=y
        # for lb in range (10,0,-10):
        #     for lg in range (10,0,-10):
        lb=lg=0
        lr=30
        # for lr in range (200,20,-10):
        for ub in range (50,100,10):
            for ug in range (50,100,10):
                for ur in range (100,255,10):
                    pinHist = 0
                    specHist = 0
                    for i in range(0,10):
                        mask = cv2.inRange(img_rgb,np.array([lb,lg,lr]),np.array([ub,ug,ur]))
                        output = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
                        crop.append(output[pin_crop_ranges[i][0]+y:pin_crop_ranges[i][1]+y1,pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
                        hist = cv2.calcHist([crop[i]],[2],None,[4], [10,50])
                        sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
                        pinHist = pinHist + sumHist[i]
                        if isGreater(sumHist[i],i):
                            print (i,lb,lg,lr,ub,ug,ur, sumHist[i], hist[0], hist[1], hist[2], hist[3])
                    for i in range(0,3):
                        scrop.append(output[scrop_ranges[i][0]+y:scrop_ranges[i][1]+y1,scrop_ranges[i][2]+x:scrop_ranges[i][3]+x1])
                        hist = cv2.calcHist([scrop[i]],[2],None,[4], [10,50])
                        sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
                        specHist = specHist + sumHist[i]
                        if isGreaterSpecial(sumHist[i],i):
                            print ('s',i,lb,lg,lr,ub,ug,ur, sumHist[i], hist[0], hist[1], hist[2], hist[3])
                    if (pinHist/10-specHist/3>oldHist):
                        oldHist = pinHist/10-specHist/3
                        print('New Max',lb,lg,lr,ub,ug,ur, oldHist)
                        olb = lb
                        olg = lg
                        olr = lr
                        oub = ub
                        oug = ug
                        our = ur
                    print('Status',lb,lg,lr,ub,ug, olb, olg, olr,oub,oug,our, oldHist)    
                    crop = []
                    scrop = []
        for i in range(0,10):
            mask = cv2.inRange(img,np.array([olb,olg,olr]),np.array([oub,oug,our]))
            output = cv2.bitwise_and(img, img, mask=mask)
            crop.append(output[pin_crop_ranges[i][0]+y:pin_crop_ranges[i][1]+y1,pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
            hist = cv2.calcHist([crop[i]],[2],None,[4], [10,50])
            sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
            pinHist = pinHist + sumHist[i]

            print (i,olb,olg,olr,oub,oug, our, sumHist[i], hist[0], hist[1], hist[2], hist[3])
        for i in range(0,3):
            scrop.append(output[scrop_ranges[i][0]+y:scrop_ranges[i][1]+y1,scrop_ranges[i][2]+x:scrop_ranges[i][3]+x1])
            hist = cv2.calcHist([scrop[i]],[2],None,[4], [10,50])
            sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
            specHist = specHist + sumHist[i]

            print ('s',i,olb,olg,olr,oub,oug,our, sumHist[i], hist[0], hist[1], hist[2], hist[3])
        # print('Status2',lb,lg,lr,ub)                      

img = cv2.imread('C:/Users/cliff/pictures/BArmMask.jpg',1)
findPins(img)
drawPinRectangles(img)
simg = pimg = img



cv2.imshow('ddd',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

