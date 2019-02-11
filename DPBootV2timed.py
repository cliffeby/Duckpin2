# import the necessary packages

import time
from datetime import datetime
import cv2
import numpy
import RPi.GPIO as GPIO
import Iothub_client_functions as iot
import picamera
import io
import threading
import cropdata1440
import myGPIO
from picamera.array import PiRGBArray
import picamera.array
from PIL import Image

pinsGPIO = myGPIO.pinsGPIO
sensor = myGPIO.sensor
segment7s = myGPIO.segment7s
segment7All = myGPIO.segment7All
pin_crop_ranges = cropdata1440.pin_crop_ranges

def setResolution():
    resX = 1440  #640
    resY = 900  #480
    res = (int(resX), int(resY))
    return res

def getCroppedImage(image,crop_array):
    croppedImage = image[crop_array[0]:crop_array[1],crop_array[2]:crop_array[3]]
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

def lightsOFF(pins):
    for pin in pins:
        GPIO.output(pin, GPIO.HIGH)
        
def tripSet():
    global sensor
    for s in sensor:
        GPIO.setup(s, GPIO.OUT)
        GPIO.output(s, GPIO.LOW)
        time.sleep(.5)
        GPIO.setup(s, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

def getMaskFrame():
    global mask_gray, resetArmCrops, ballCrops, img_gray1arm
    frame1 = cv2.imread('/home/pi/Shared/histImage/BallMask.jpg',1)
    img_arm = getCroppedImage(frame1, resetArmCrops)
    (h,w,d) = img_arm.shape
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

def timeout():
        global timesup
        timesup = True
        print ('Pin timer is finished', timesup)

def timeoutDeadwood():
        global timesupDeadwood
        timesupDeadwood = True
        print ('Deadwood timer is finished', timesupDeadwood)

def timeoutReset():
        global timesupReset
        timesupReset = True
        print ('Deadwood timer is finished', timesupReset)

def findPins():
        global x,x1,y,y1
        global priorPinCount, frameNo
        global img_rgb
        global frame2
        global pinsFalling, timesup  # initial values False, True
        
        pinCount = 0
        crop = []
        sumHist = [0,0,0,0,0,0,0,0,0,0]
        lower_red = numpy.array([0,0,70]) # lower_red = np.array([0,100,0])
        upper_red = numpy.array([110, 110, 255])  # upper_red = np.array([180,255,255])

        mask = cv2.inRange(img_rgb,lower_red,upper_red)
        output = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        threshold1 = 10
        for i in range(0,10):
                crop.append(output[pin_crop_ranges[i][0]+y:pin_crop_ranges[i][1]+y1,pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
                hist = cv2.calcHist([crop[i]],[1],None,[4], [10,50])
                sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
                # print (i, sumHist[i])
                if threshold1 < sumHist[i]:
                    pinCount = pinCount + 2**(9-i)

        bit_GPIO(pinsGPIO,pinCount)
        if pinsFalling == True:
                if timesup == False:
                    return
                else:
                    result = " _"+ str(priorPinCount)+"_" + str(pinCount) + "_" +str(frameNo)
                    print("FrameNo ", frameNo, "PinCount ", priorPinCount, "_",pinCount, result )
                    if priorPinCount == 1023:
                        write_video(stream, result)
                    priorPinCount = pinCount
                    pinsFalling = False
                    return
        if priorPinCount <= pinCount: 
            priorPinCount = pinCount
            return
        else:
            pinsFalling = True
            t = threading.Timer(2.0, timeout)
            timesup = False
            t.start() # after 2.0 seconds, stream will be saved
            print ('timer is running', priorPinCount, pinCount)
            return

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

def iotSendImg(buf):
    global frameNo
    try:
        client = iot.iothub_client_init()
        # if client.protocol == IoTHubTransportProvider.MQTT:
        print ( "IoTHubClient is reporting state" )
        reported_state = "{\"newState\":\"standBy\"}"
        # td = datetime.now()
        client.send_reported_state(reported_state, len(reported_state), iot.send_reported_state_callback, iot.SEND_REPORTED_STATE_CONTEXT)
        filename = "imgdp" + time.strftime('%A') + ".jpg"
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

    for i in range(0,10):
        a =(pin_crop_ranges[i][2]+mx,pin_crop_ranges[i][0]+my)
        b = (pin_crop_ranges[i][3]+mx, pin_crop_ranges[i][1]+my)
        cv2.rectangle(ball_image, b, a, 255, 2)
        if i == 6:
            cv2.putText(ball_image,str(a),a,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
            cv2.putText(ball_image,str(b),(b[0]-250,b[1]),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
    cv2.imwrite('/home/pi/Shared/videos/CCEPinMask'+str(i) +'.jpg',ball_image)
    iotSendImg('/home/pi/Shared/videos/CCEPinMask9.jpg')
    
setupGPIO(pinsGPIO)
setupGPIO(segment7s)
tripSet()
priorPinCount = 0
pinsFalling = False
timesup = True
activity = "\r\n"
x=33
x1=0 +x
y=-15
y1=0 + y
frameNo = 0
ballCounter = 0
videoReadyFrameNo = 10
timesupDeadwood = True
timesupReset = True
# deadwoodTimer = time.time()
lightsOFF(segment7s)
GPIO.output((segment7All[0]), GPIO.LOW)

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
    print(camera.resolution)

    for frame in camera.capture_continuous(rawCapture,format="bgr",  use_video_port=True):
        # grab the raw NumPy array representing the image, then initialize the timestamp
        # and occupied/unoccupied text???????????????????
        rawCapture.truncate()
        rawCapture.seek(0)
        
        frame2 = frame.array
        frameNo = frameNo +1
        img_rgb = frame2

        while (GPIO.input(sensor[0]) == GPIO.HIGH):
            GPIO.wait_for_edge(sensor[0], GPIO.FALLING)
            print('done')
            time.sleep(.05)
            if GPIO.input(sensor[0]) == 0:
                ballCounter= ballCounter+1
                print ("Ball Falling edge", ballCounter)
                lightsOFF(segment7s)
                GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
                print('Ball Timer Awake ', ballCounter)               
        if (GPIO.input(sensor[1]) == GPIO.HIGH):
                print('Deadwood ', ballCounter)
                if timesupDeadwood == True:
                    tDeadwood = threading.Timer(10.0, timeoutDeadwood)
                    timesupDeadwood = False
                    tDeadwood.start() # after x seconds, stream will be saved
                    print ('Deadwood timer is running', ballCounter)
        while (GPIO.input(sensor[2]) == GPIO.HIGH):
                print('Reset -pre', ballCounter)
                ballCounter = 0
                lightsOFF(segment7s)
                GPIO.output((segment7All[0]), GPIO.LOW)
                bit_GPIO(pinsGPIO,1023)
                GPIO.wait_for_edge(sensor[0], GPIO.FALLING)
                if timesupReset == True:
                    tReset = threading.Timer(3.0, timeoutReset)
                    timesupReset = False
                    tReset.start() # after 3.0 seconds, stream will be saved
                    print ('Reset timer is running', ballCounter)

        writeImageSeries(30, 1, img_rgb)
        # if deadwoodTimer+10<time.time():
        #     print(deadwoodTimer, time.time())
        if frameNo%4== 0:
            if timesup ==True and timesupDeadwood ==True and timesupReset== True:
                findPins()
        # else:
        #     print('Skipped findPins()')
