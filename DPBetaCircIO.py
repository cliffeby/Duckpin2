# import the necessary packages

import time
import cv2
import numpy
import RPi.GPIO as GPIO
import Iothub_client_functions as iot
import picamera
import io
# from picamera import PiCamera, Color
from picamera.array import PiRGBArray
import picamera.array
from PIL import Image

pinsGPIO = [15,14,3,2,21,20,16,5,26,6]
# mask_crop_ranges = ([1100,1700, 220,2800],[0,0,0,0])
pin_crop_ranges = ([445,515, 755,825],[360,440, 710,755],[370,450, 885,940],[320,385, 655,705],[320,385, 815,875],
    [330,380, 980,1035],[275,345, 605,665],[275,345, 745,805],[275,345, 895,955],[275,345, 1060,1120])

def setResolution():
    resX = 640
    resY = 480
    res = (int(resX), int(resY))
    return res
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

def writeImageSeries(frameNoStart, numberOfFrames, img_rgb):
    if frameNoStart <= frameNo:
        if frameNo <= frameNoStart+numberOfFrames:
            print ('Saving ../home/pi/Shared/videos/videoCCEFrame'+ str(frameNo) +'.jpg')
            cv2.imwrite('/home/pi/Shared/videos/videoCCEFrame'+ str(frameNo) +'.jpg',img_rgb)

def write_video(stream):
# Write the entire content of the circular buffer to disk. No need to
# lock the stream here as we're definitely not writing to it
# simultaneously
    global motion_filename, frameNo, videoReadyFrameNo
    
    if frameNo<videoReadyFrameNo+120:
        return
    videoReadyFrameNo = frameNo
    print("writng1 ",motion_filename)
    with io.open('/tmp/ramdisk/myfile.h264', 'wb') as output:
        for frame in stream.frames:
            if frame.frame_type == picamera.PiVideoFrameType.sps_header:
                stream.seek(frame.position)
                break
        while True:
            buf = stream.read1()
            
            if not buf:
                break
           
        
            output.write(buf)
    iotSend('/tmp/ramdisk/myfile.h264')
            
    # Wipe the circular stream once we're done
    stream.seek(0)
    stream.truncate()

def isPinSetter():
    global setterPresent
    global frameNo
    global img_rgb
    global firstSetterFrame
    global activity
    
    # Convert BGR to HSV
    # frame = img_rgb[150:450, 650:1600]
    frame = img_rgb[75:225, 325:800]
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
    if area >1000:
        setterPresent = True
        firstSetterFrame = frameNo
    if setterPresent:
        activity = activity + str(priorPinCount)+ ',-2,'
        print("Green", area, frameNo)
    else:
        firstSetterFrame = 0

def isResetArm():
    global firstArmFrame, armPresent, ballCounter
    global frameNo
    global img_rgb
    global img_gray1arm
    global threshArm
    frame2arm = img_rgb[155:270,540:600]
    img_gray2arm = cv2.cvtColor(frame2arm, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(img_gray1arm,img_gray2arm)
    # First value reduces noise.  Values above 150 seem to miss certain ball colors
    ret, threshArm = cv2.threshold(diff, 120,255,cv2.THRESH_BINARY)
    frame = threshArm
    # Blur eliminates noise by averaging surrounding pixels.  Value is array size of blur and MUST BE ODD
    threshArm = cv2.medianBlur(threshArm,15)
    cnts = cv2.findContours(threshArm.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    radius = 0
    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((xContour, yContour), radius) = cv2.minEnclosingCircle(c)
        if radius>15:
            print('Reset Arm', radius, frameNo, len(cnts))
            firstArmFrame = frameNo
            armPresent = True
            ballCounter = 0

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
        lower_red = numpy.array([0,0,70]) # lower_red = np.array([0,100,0])
        upper_red = numpy.array([110, 110, 255])  # upper_red = np.array([180,255,255])

        mask = cv2.inRange(img_rgb,lower_red,upper_red)
        output = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        threshold1 = 20
        threshold2 = 20
        for i in range(0,6):
                crop.append(output[pin_crop_ranges[i][0]+y:pin_crop_ranges[i][1]+y1,pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
                hist = cv2.calcHist([crop[i]],[1],None,[4], [10,50])
                sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
                
                    
                print (i, sumHist[i])
                if threshold1 < sumHist[i]:
                    pinCount = pinCount + 2**(9-i)
        for i in range(6,10):
                crop.append(output[pin_crop_ranges[i][0]+y:pin_crop_ranges[i][1]+y1,pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
                hist = cv2.calcHist([crop[i]],[1],None,[4], [10,50])
                sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
                
                    
                print (i, sumHist[i])
                if threshold2 < sumHist[i]:
                    pinCount = pinCount + 2**(9-i)
                
        print('HIST', frameNo, pinCount)
        bit_GPIO(pinsGPIO,pinCount)

        if priorPinCount == pinCount:
            return False
        else:
            write_video(stream)
            priorPinCount = pinCount
            return True

def iotSend(buf):
    global frameNo
    try:
        client = iot.iothub_client_init()
        # if client.protocol == IoTHubTransportProvider.MQTT:
        print ( "IoTHubClient is reporting state" )
        reported_state = "{\"newState\":\"standBy\"}"
        client.send_reported_state(reported_state, len(reported_state), iot.send_reported_state_callback, iot.SEND_REPORTED_STATE_CONTEXT)
        filename = "vidforblob" + str(frameNo) + ".h264"
        f = open(buf, "rb+")
        content = f.read()
        
        print("CONTENT LEN", len(content), type(content))
        client.upload_blob_async(filename,content, len(content), iot.blob_upload_conf_callback,1001)


    except iot.IoTHubError as iothub_error:
        print ( "Unexpected error %s from IoTHub" % iothub_error )
        return
    except KeyboardInterrupt:
        print ( "IoTHubClient sample stopped" )

    iot.print_last_message_time(client)    

setupGPIO(pinsGPIO)
setterPresent = False
armPresent = False
maskFrame = True
priorPinCount = 0
activity = "\r\n"
x=-50
x1=0 +x
y=-165
y1=0 + y
crop_ranges = ([500,900,100,1240],[0,0,0,0])
# crop_ranges = ([600,900, 110,1400],[0,0,0,0])
ballCoords=[0]*100
frameNo = 0
prevFrame = 0
ballCounter = 0
ballCounterFrame = 0
videoReadyFrameNo = 0
origCounter = 0
pinReactionTime = 0
pinReactionFlag = False
video_preseconds = 3
motion_width = 640
motion_height = 480
motion_filename = "DPBetaCIOTest"
for i in range(0,1):
    a =(int(crop_ranges[i][2]/2)+x,int(crop_ranges[i][0]/2)+y)
    b = (int(crop_ranges[i][3]/2)+x1, int(crop_ranges[i][1]/2)+y1)
with picamera.PiCamera() as camera:
    camera.resolution = setResolution()
    camera.framerate = 25
    camera.video_stabilization = True
    camera.annotate_background = True
    camera.rotation = 180
    rawCapture = PiRGBArray(camera, size=camera.resolution)
        # setup a circular buffer
    # stream = picamera.PiCameraCircularIO(camera, seconds = video_preseconds)
    stream = picamera.PiCameraCircularIO(camera, size = 1000000)
    # video recording into circular buffer from splitter port 1
    camera.start_recording(stream, format='h264', splitter_port=1)
        #camera.start_recording('test.h264', splitter_port=1)
        # low resolution motion vector analysis from splitter port 2
    camera.start_recording('/dev/null', splitter_port=2, resize=(motion_width,motion_height) ,format='h264')
        # wait some seconds for stable video data
    camera.wait_recording(2, splitter_port=1)
    motion_detected = False

    print(camera.resolution)
    time.sleep(1)
    for frame in camera.capture_continuous(rawCapture,format="bgr",  use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text
        rawCapture.truncate()
        rawCapture.seek(0)
        
        frame2 = frame.array
        if maskFrame:
            frame1 = frame.array
            # mask= frame1[650:900, 250:1500]
            # mask= frame1[500:900, 100:1240]
            mask = frame1[250:470, 50:600]
            frame1 = mask
            maskFrame = False
            continue
        frameNo = frameNo +1
        img_rgb = frame2
        frame1arm = frame2[155:270,540:600]
        img_gray1arm = cv2.cvtColor(frame1arm, cv2.COLOR_BGR2GRAY)
        # cv2.imwrite('../videos/videoCCEFrame'+ str(frameNo) +'.jpg',img_rgb)
        # if pinReactionFlag:
        #     if time.process_time()-3 > pinReactionTime:
                
        #         activity = activity + str(priorPinCount)+','
        #         print(activity)
        #         pinReactionFlag = False

        # # if setterPresent:
        # #         if firstSetterFrame + 60 > frameNo:
        # #             continue
        # if armPresent:
        #         if firstArmFrame + 70 > frameNo:
        #             continue
        #         if firstArmFrame+ 70 == frameNo:
        #             armPresent = False
        #             activity = activity + str(priorPinCount)+ ',-1,'
        if setterPresent:
                if firstSetterFrame + 120 > frameNo:
                    continue
        if armPresent:
            if firstArmFrame + 120 > frameNo:
                continue
            if firstArmFrame+ 120 == frameNo:
                ballCounter = 0
                armPresent = False
        if setterPresent or armPresent:
            continue
        isPinSetter()
        isResetArm()
   
        # frame2= frame2[320:480,40:565]
       
        # frame2= frame2[650:900, 250:1500]
        frame2= frame2[250:470, 50:600]
        img_gray1 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(img_gray1,img_gray2)
        # First value reduces noise.  Values above 150 seem to miss certain ball colors
        ret, thresh = cv2.threshold(diff, 120,255,cv2.THRESH_BINARY)
        frame = thresh
        # Blur eliminates noise by averaging surrounding pixels.  Value is array size of blur and MUST BE ODD
        thresh = cv2.medianBlur(thresh,13)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        radius = 0
        if len(cnts) > 0:
                # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and centroid
            c = max(cnts, key=cv2.contourArea)
            # ((xContour, yContour), radius) = cv2.minEnclosingCircle(c)
            print('Ball Area', frameNo, len(cnts))
            if prevFrame + 15 < frameNo:
                    ballCounter = ballCounter + 1
                    print("BALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL", ballCounter)
                    prevFrame = frameNo
            # only proceed if the radius meets a minimum size
            # if radius > 5:
            #     # draw the circle and centroid on the frame,
            #     # then update the list of tracked points
            #     M = cv2.moments(c)
            #     center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            #     cv2.drawContours(img_gray2, cnts, -1, (0,255,0), 3)
            #     if prevFrame + 15 < frameNo:
            #         ballCounter = ballCounter + 1
            #         prevFrame = frameNo
            #     print('BALL CENTER',center, radius, frameNo, len(cnts), ballCounter)
                
                        # cv2.imwrite('P:videos/cv2Img'+str(frameNo)+'.jpg',img_gray2)
        img_gray1=img_gray2        
        cv2.imshow('Ball', img_gray2)
        cv2.imshow('Arm', threshArm)
        # cv2.imshow('Thresh' , thresh)
        camera.annotate_text = "Date "+ str(time.process_time()) + " Frame " + str(frameNo) + " Prior " + str(priorPinCount)
        # writeImageSeries(20, 3, img_rgb)
        # cv2.imshow('Frame' , img_rgb)
        if frameNo%5 ==0:
            tf = findPins()        

        # cv2.rectangle(img_rgb,b, a, 255,2)

        # cv2.imshow('IMG_RGB with Ball Rect', img_rgb)
        # writeImageSeries(135,20)
        
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
