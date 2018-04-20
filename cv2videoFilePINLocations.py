# import the necessary packages

import time
import cv2
import numpy
from matplotlib import pyplot as plt

def show_color_histogram(image):
    # for i, col in enumerate(['b', 'g', 'r']):
    #     draw_image_histogram(image, [i], color=col)
    # plt.show()
    draw_image_histogram(image, [0])

def with_open_cv(image):
    img_new = image[:,:,[0,2]] = 0
    cv2.imshow('result.jpg',img_new)

def draw_image_histogram(image, channels):
    hist = cv2.calcHist([image], channels, None, [3], [10, 250])
    
    # plt.plot(hist, color=color)
    # plt.xlim([0, 10])
    print('Green', hist, frameNo)

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

def writeImageSeries(frameNoStart, numberOfFrames):
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
        #Contour area is taken
        area = cv2.contourArea(cnt) +area
    if area >1000:
        setterPresent = True
        firstSetterFrame = frameNo
    if setterPresent:
        print("Green", area, frameNo)
    else:
        firstSetterFrame = 0
    cv2.imshow('frame',frame)

    
# initialize the camera and grab a reference to the raw camera capture
# cap = cv2.VideoCapture('../feb28/bffl28.h264')
cap = cv2.VideoCapture('../videos/video3d.h264')
setterPresent = False
PBFN = 0
PBFD= []
newSeries = True
x=-0
x1=0 +x
y=-0
y1=0 + y

lower_red = numpy.array([0,0,100]) # lower_red = np.array([0,100,0])
upper_red = numpy.array([110, 110, 255])  # upper_red = np.array([180,255,255])
crop_ranges = ([1100,1700, 220,2800],[0,0,0,0])

pts = []
kernel = numpy.ones((5,5), numpy.uint8)
frameNo = 0
prevFrame = 0
ballCounter = [0]*3
origCounter = 0
for i in range(0,1):
    a =(int(crop_ranges[i][2]/2)+x,int(crop_ranges[i][0]/2)+y)
    b = (int(crop_ranges[i][3]/2)+x1, int(crop_ranges[i][1]/2)+y1)
ret, frame1 = cap.read()
# frame1= frame1[1100:220, 1700:2850]
mask= frame1[550:850, 250:1600]

frame1 = mask
while(cap.isOpened()):
    ret, frame2 = cap.read()
    frameNo = frameNo +1
    img_rgb = frame2
    
    if setterPresent:
            if firstSetterFrame + 34 > frameNo:
                continue    
    isPinSetter()
    frame2= frame2[550:850, 250:1600]
    hist = cv2.calcHist(frame2,[1],None,[4], [10,50])
    
    img_gray1 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(img_gray1,img_gray2)
    ret, thresh = cv2.threshold(diff, 100,200,cv2.THRESH_BINARY)
    frame = thresh
    # thresh = cv2.erode(thresh, kernel, iterations=1)
    # thresh = cv2.dilate(thresh, kernel, iterations=1)
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
       
        
        ballCounter[0]=0
        ballCounter[1]=0
        ballCounter[2]=0
		# only proceed if the radius meets a minimum size
        if radius > 30:
         if radius < 120 :
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.circle(img_gray2, center, int(radius), (0, 0, 255), -1)
            if center < (1100,200):
                # if newSeries==True:
                #     prevFrame = frameNo-1
                #     newSeries = False
                #     PBFN = 0
                # if isFrameNext(frameNo, prevFrame):
                #     prevFrame = frameNo
                #     ballOrReset(frame1,center)

                    print('CENTER',center, radius, frameNo, len(cnts))
                    cv2.imwrite('P:videos/cv2Img'+str(frameNo)+'.jpg',img_gray2)
                # else:
                    print('Final',frameNo, PBFD)
                    newSeries = True
	# update the points queue
        # pts.append(center)
        # # loop over the set of tracked points
        # for i in range(1, len(pts)):
        #     # if either of the tracked points are None, ignore
        #     # them
        #     if pts[i - 1] is None or pts[i] is None:
        #         continue

        #     # otherwise, compute the thickness of the line and
        #     # draw the connecting lines
        #     # thickness = int(numpy.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        #     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), 1)
        #     # if pts[i][0] < 1100:
        #         # print('PTS',pts[i])

        # # show the frame to our screen
        cv2.imshow("Frame", img_rgb)
    cv2.rectangle(img_rgb,b, a, 255,2)
    # cv2.imshow("Frame1/Mask", frame1)
    # cv2.imshow('BO.jpg', img_gray1)
    cv2.imshow('Diff', thresh)
    

    # writeImageSeries(135,20)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream in preparation for the next frame
    # frame.truncate(0)
    # mask = frame1
    # frame1=frame2
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break