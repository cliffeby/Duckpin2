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

def findPins(img_rgb, img_rgb1):        
        global priorPinCount, frameNo
        global crop_ranges, scrop_ranges,pin_crop_ranges, sumHist
        global x,y,x1,y1,crop
        hist =[]
        Shist = np.zeros((10,4,1))
        pin_crop_ranges = crop_ranges
        pinCount = 0
        crop = []
        crope = []
        scrop = []
        sumHist = [0,0,0,0,0,0,0,0,0,0]
        lower_red = np.array([0,0,50]) # lower_red = np.array([0,100,0])  try 0,0,50
        upper_red = np.array([150, 150, 240])  # upper_red = np.array([180,255,255])
        pinHist=specHist=0
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

        for i in range(0,10):
            for k in range(0,10):
                # mask = cv2.inRange(img_rgb,np.array([0,0,70]),np.array([110,110,255]))
                # output = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
                output = img_rgb
                output1 = img_rgb1
                crop.append(output[pin_crop_ranges[i][0]+y:pin_crop_ranges[i][1]+y1,pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
                hist = cv2.calcHist([crop[i]],[1],None,[4], [0,255])
                

                
                Shist = {0:hist}
                crope.append(output1[pin_crop_ranges[k][0]+y:pin_crop_ranges[k][1]+y1,pin_crop_ranges[k][2]+x:pin_crop_ranges[k][3]+x1])
                hists = cv2.calcHist([crope[k]],[1],None,[4], [0,255])           
                
                
                d= cv2.compareHist(Shist[0], hists,0)
                print (i,k,d,  hists, Shist)


img = cv2.imread('/home/cliffeby/Downloads/imgdpMonday.jpg',1)
imge = cv2.imread('/home/cliffeby/Downloads/imgdpThursday.jpg',1)
findPins(img, imge)
drawPinRectangles(imge)

# cv2.imshow('ddd',imge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

