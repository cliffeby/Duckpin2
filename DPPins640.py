# import the necessary packages

import time
from datetime import datetime
import cv2
import numpy, statistics, collections
import RPi.GPIO as GPIO
import Iothub_client_functions as iot
import picamera
import io, threading
import cropdata640
import myGPIO
from picamera.array import PiRGBArray
import picamera.array
from PIL import Image

pinsGPIO = myGPIO.pinsGPIO
segment7s = myGPIO.segment7s
segment7All = myGPIO.segment7All
pin_crop_ranges = cropdata640.pin_crop_ranges
resetArmCrops = cropdata640.resetArmCrops
pinSetterCrops = cropdata640.pinSetterCrops
ballCrops = cropdata640.ballCrops

def setResolution():
    resX = 640
    resY = 480
    res = (int(resX), int(resY))
    return res

def getCroppedImage(image,crop_array):
    croppedImage = image[crop_array[0]:crop_array[1],crop_array[2]:crop_array[3], :]
    return croppedImage

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

def getMaskFrame():
    global mask_gray, resetArmCrops, ballCrops, img_gray1arm
    frame1 = cv2.imread('/home/pi/Shared/histImage/BallMask640.jpg',1)
    img_arm = getCroppedImage(frame1, resetArmCrops)
    # (h,w,d) = img_arm.shape
    img_gray1arm = cv2.cvtColor(img_arm, cv2.COLOR_BGR2GRAY)
    maskFrame = getCroppedImage(frame1, ballCrops)
    mask_gray = cv2.cvtColor(maskFrame, cv2.COLOR_BGR2GRAY)
            
def writeImageSeries(frameNoStart, numberOfFrames, img_rgb):
    if frameNoStart <= frameNo:
        if frameNo <= frameNoStart+numberOfFrames:
            print ('Saving ../home/pi/Shared/videos/videoCCEFrame'+ str(frameNo) +'.jpg')
            cv2.imwrite('/home/pi/Shared/videos/videoCCEFrame'+ str(frameNo) +'.jpg',img_rgb)
            drawPinRectangles()

def write_video(stream,result):
# Write the entire content of the circular buffer to disk. No need to
# lock the stream here as we're definitely not writing to it
# simultaneously
    global frameNo, videoReadyFrameNo
    if frameNo < videoReadyFrameNo + 120:
        return
    videoReadyFrameNo = frameNo
    print("writng dp ", result)
    #setup ram dsk


    with io.open('/dp/log/firstFile.h264', 'wb') as output:
        for frame in stream.frames:
            if frame.frame_type == picamera.PiVideoFrameType.sps_header:
                stream.seek(frame.position)
                break
        while True:
            buf = stream.read1()
            if not buf:
                break
            output.write(buf)
    iotSend('/dp/log/firstFile.h264',result)
            
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
    frame =  getCroppedImage(img_rgb, pinSetterCrops)
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
    if setterPresent:
        activity = activity + str(priorPinCount)+ ',-2,'
        print("Green", area, frameNo, activity)
    return

def isResetArm():
    global firstArmFrame, armPresent, ballCounter
    global frameNo
    global img_rgb
    global img_gray1arm
    global threshArm
    global resetArmCrops
    global priorPinCount
    
    frame2arm = getCroppedImage(img_rgb, resetArmCrops)
    img_gray2arm = cv2.cvtColor(frame2arm, cv2.COLOR_BGR2GRAY)
    # print('IMG GRAY ARM', img_gray1arm, img_gray2arm, frame2arm, type(frame2arm))
    # print(type(img_gray1arm), type(img_gray2arm))
    diff = cv2.absdiff(img_gray1arm,img_gray2arm)
    # First value reduces noise.  Values above 150 seem to miss certain ball colors
    ret, threshArm = cv2.threshold(diff, 150,255,cv2.THRESH_BINARY)
    if cv2.countNonZero(threshArm)>300:
        print("Non Zero Setter............", frameNo, cv2.countNonZero(threshArm))
    # frame = threshArm
    # Blur eliminates noise by averaging surrounding pixels.  Value is array size of blur and MUST BE ODD
    threshArm = cv2.medianBlur(threshArm,15)
    # cv2.imshow('arm trhesh', threshArm)
    cnts = cv2.findContours(threshArm.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]

    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((xContour, yContour), radius) = cv2.minEnclosingCircle(c)
        if radius > 2:
            print('Reset Arm', frameNo, len(cnts), ballCounter, " ", priorPinCount)
            armPresent = True
            ballCounter = 0
            bit_GPIO(pinsGPIO,1023)
    return

def findPins():
        global x,x1,y,y1
        global priorPinCount, frameNo
        global img_rgb
        global frame2
        global pinsFalling, timesup  # initial values False, True
        def timeout():
            global timesup
            timesup = True
            print ('Timer is finished')
        pinCount = 0
        crop = []
        sumHist = [0,0,0,0,0,0,0,0,0,0]
        lower_red = numpy.array([0,180,80])
        # upper_red = numpy.array([10,255,255])  
        upper_red = numpy.array([10,220,190])       
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv,lower_red,upper_red)
        output = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
        threshold = 1
        for i in range(0,10):
                crop.append(output[pin_crop_ranges[i][0]+y:pin_crop_ranges[i][1]+y1,pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
                hist = cv2.calcHist([crop[i]],[1],None,[4], [1,255])
                sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
                # print (i, sumHist[i])
                if threshold < sumHist[i]:
                    pinCount = pinCount + myModeFilter(i)


        bit_GPIO(pinsGPIO,pinCount)
        if pinsFalling == True:
                if timesup == False:
                    return
                else:
                    result = " _"+ str(priorPinCount)+"_" + str(pinCount) + "_" +str(frameNo)
                    # print("FrameNo ", frameNo, "PinCount ", priorPinCount, "_",pinCount, result )
                    # if priorPinCount == 1023:
                    #     write_video(stream, result)
                    priorPinCount = pinCount
                    pinsFalling = False
                    return
        if priorPinCount <= pinCount: 
            priorPinCount = pinCount
            return
        else:
            pinsFalling = True
            # t = threading.Timer(3.0, timeout)
            # timesup = False
            # t.start() # after 2.0 seconds, stream will be saved
            # # print ('timer is running', priorPinCount, pinCount)
            time.sleep(2)
            return

def myModeFilter(index):
    global pinCounts
    pinCounts[index].append(1)
    newValue = statistics.mode(pinCounts[index])
    if newValue ==1:
        return 2**(9-index)
    else:
        return 0

def resetResetVars():
    global armPresent,ballPresent,ballCounter,tArmStart
    armPresent = False
    ballPresent = False
    ballCounter = 0
    tArmStart = time.time()
    print('ResetArmVars')

def iotSend(buf, result):
    global frameNo
    try:
        client = iot.iothub_client_init()
        # if client.protocol == IoTHubTransportProvider.MQTT:
        print ( "IoTHubClient is reporting state" )
        reported_state = "{\"newState\":\"standBy\"}"
        # td = datetime.now()
        client.send_reported_state(reported_state, len(reported_state), iot.send_reported_state_callback, iot.SEND_REPORTED_STATE_CONTEXT)
        filename = "dp" + result + ".h264"
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

def drawPinRectangles():
    global ball_image,img_rgb,x,y
    global pin_crop_ranges
    mx=x
    my=y
    ball_image = img_rgb
    # NOTE: crop is img[y: y + h, x: x + w] 
    # cv2.rectangle is a = (x,y) , b=(x1,y1)

    for i in range(0,9):
        a =(pin_crop_ranges[i][2]+mx,pin_crop_ranges[i][0]+my)
        b = (pin_crop_ranges[i][3]+mx, pin_crop_ranges[i][1]+my)
        cv2.rectangle(ball_image, b, a, 255, 2)
        if i == 6:
            cv2.putText(ball_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(ball_image,str(b),(b[0]-250,b[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imwrite('/home/pi/Shared/videos/CCEPinMask'+str(i) +'.jpg',ball_image)
    a = (ballCrops[2]+mx,ballCrops[0]+my)
    b = (ballCrops[3]+mx, ballCrops[1]+my)
    cv2.rectangle(ball_image, b, a, 255, 2)
    cv2.putText(ball_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(ball_image,str(b),(b[0]-250,b[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imwrite('/home/pi/Shared/videos/CCEBBallMask'+str(i) +'.jpg',ball_image)
    a = (resetArmCrops[2]+mx, resetArmCrops[0]+my)
    b = (resetArmCrops[3]+mx, resetArmCrops[1]+my)
    cv2.rectangle(ball_image, b, a, 255, 2)
    cv2.putText(ball_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.putText(ball_image,str(b),(b[0]-250,b[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imwrite('/home/pi/Shared/videos/CCEBBallMask'+str(i) +'.jpg',ball_image)


def lightsOFF(pins):
    for pin in pins:
        GPIO.output(pin, GPIO.HIGH)

setupGPIO(pinsGPIO)
setupGPIO(segment7s)
# setupGPIO(sensor)
lightsOFF(segment7s)
GPIO.output((segment7All[0]), GPIO.LOW)
getMaskFrame()
bitBuckets =  collections.deque(13*[1], 13)
pinCounts =[bitBuckets for x in range(10)]
setterPresent = False
armPresent = False
ballPresentFrameNo = 0
priorPinCount = 0
pinsFalling = False
timesup = True
activity = "\r\n"
x=0
x1=0 +x
y=0
y1=0 + y
frameNo = 0
ballCounter = 0
videoReadyFrameNo = 0
video_preseconds = 3
tArmStart = time.time()

with picamera.PiCamera() as camera:
    camera.resolution = setResolution()
    camera.video_stabilization = True
    camera.annotate_background = True
    camera.rotation = 180
    rawCapture = PiRGBArray(camera, size=camera.resolution)
    # setup a circular buffer
    # stream = picamera.PiCameraCircularIO(camera, seconds = video_preseconds)
    stream = picamera.PiCameraCircularIO(camera, size = 3000000)
    # video recording into circular buffer from splitter port 1
    camera.start_recording(stream, format='h264', splitter_port=1)
    #camera.start_recording('test.h264', splitter_port=1)
    # wait 2 seconds for stable video data
    camera.wait_recording(2, splitter_port=1)
    # motion_detected = False
    print(camera.resolution)

    for frame in camera.capture_continuous(rawCapture,format="bgr",  use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text???????????????????
        rawCapture.truncate()
        rawCapture.seek(0)
        
        frame2 = frame.array
        frameNo = frameNo +1
        img_rgb = frame2
        frame2arm = getCroppedImage(frame2, resetArmCrops)
        img_gray2arm = cv2.cvtColor(frame2arm, cv2.COLOR_BGR2GRAY)

        isPinSetter()   #Deadwood
        if setterPresent:
            print('SetterPresent', frameNo, ballCounter)
            bit_GPIO(pinsGPIO,priorPinCount)
            time.sleep(7)
            setterPresent = False
            ballPresent = False
            continue
        
        if armPresent == False:
            isResetArm()    #Reset
        else:
            # if time.time()-tArmStart >10:
                time.sleep(10)
                resetResetVars()
                print ('ArmPresent', frameNo, ballCounter) 
                continue
        
        frame2= getCroppedImage(frame2, ballCrops)
    
        # img_gray1 = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(mask_gray,img_gray)
        # First value reduces noise.  Values above 150 seem to miss certain ball colors
        ret, thresh = cv2.threshold(diff, 120,255,cv2.THRESH_BINARY)
        frame = thresh
        
        # Blur eliminates noise by averaging surrounding pixels.  Value is array size of blur and MUST BE ODD
        thresh = cv2.medianBlur(thresh,31)
        nzb = cv2.countNonZero(thresh)
        if nzb>900:
            print("Non Zero Ball.............", frameNo, nzb)
        # print(type(thresh), type(diff),type(img_gray1), type(img_gray2))
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None
        radius = 0
        if armPresent == False:
            if frameNo - ballPresentFrameNo >40 and nzb>800:
                # print('Here', len(cnts))
                if len(cnts) >= 0:
                    c = max(cnts, key=cv2.contourArea)
                    ((xContour, yContour), radius) = cv2.minEnclosingCircle(c)
                    # print('Rad and center',radius, center)
                if radius > 20 and radius < 40:
                    lightsOFF(segment7s)
                    ballCounter = ballCounter + 1
                    GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
                    print("BALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL", ballCounter, 'at frame ', frameNo, radius, xContour )
                    ballPresentFrameNo = frameNo

                GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
        
        # camera.annotate_text = "Date "+ str(time.process_time()) + " Frame " + str(frameNo) + " Prior " + str(priorPinCount)
        # writeImageSeries(30, 3, img_rgb)
        if frameNo%2 == 0:
            findPins()
        # cv2.imshow('Ba;;',img_gray)
        # key = cv2.waitKey(0) & 0xFF
        # # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break
