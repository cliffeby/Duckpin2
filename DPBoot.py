# import the necessary packages

import time
import credentials
from datetime import datetime
import cv2
import numpy
import RPi.GPIO as GPIO
import Iothub_client_functions as iot
import iot2
import picamera
import io
import os 
import threading
import cropdata1440
import myGPIO
import offsets
from picamera.array import PiRGBArray
import picamera.array
from PIL import Image
from azure.iot.device import IoTHubDeviceClient
from azure.core.exceptions import AzureError
from azure.storage.blob import BlobClient

pinsGPIO = myGPIO.pinsGPIO
sensor = myGPIO.sensor
segment7s = myGPIO.segment7s
segment7All = myGPIO.segment7All
pin_crop_ranges = cropdata1440.pin_crop_ranges
CONNECTION_STRING = credentials.loginFree["ConnectionString"]

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
            print ('Saving ../home/cliffeby/Pictures/videoCCEFrame'+ str(frameNo) +'.jpg')
            cv2.imwrite('/home/cliffeby/Pictures/videoCCEFrame'+ str(frameNo) +'.jpg',img_rgb)
            drawPinRectangles()

def write_video(stream,result):
# Write the entire content of the circular buffer to disk. No need to
# lock the stream here as we're definitely not writing to it
# simultaneously
    global frameNo, videoReadyFrameNo, img_rgb, priorPinCount
    if frameNo < videoReadyFrameNo + 120:
        return
    videoReadyFrameNo = frameNo
    print("writng dp ", result)
    
    # Extract pin counts from result string
    # result format is " _beginningPinCount_endingPinCount_frameNo"
    parts = result.split('_')
    if len(parts) >= 3:
        beginning_pins = int(parts[1])
        ending_pins = int(parts[2])
        
        # Capture final frame before writing video
        if img_rgb is not None:
            captureFinalFrame(img_rgb, beginning_pins, ending_pins)
    
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
        print ('Reset timer is finished', timesupReset)

def flash():
    for i in range(1,10):
        bit_GPIO(pinsGPIO, 1023)
        lightsOFF(segment7s)
        time.sleep(.3)
        bit_GPIO(pinsGPIO,0)
        GPIO.output((segment7All[8]), GPIO.LOW)
        time.sleep(.3)
    lightsOFF(segment7All)

def findPins():
        global x,x1,y,y1
        global priorPinCount, frameNo, ballCounter
        global img_rgb
        global frame2
        global pinsFalling, timesup  # initial values False, True
        
        pinCount = 0
        crop = []
        sumHist = [0,0,0,0,0,0,0,0,0,0]
        lower_red = numpy.array([0,0,70]) # lower_red = np.array([0,100,0])
        upper_red = numpy.array([110, 110, 255])  # upper_red = n  p.array([180,255,255])

        mask = cv2.inRange(img_rgb,lower_red,upper_red)
        output = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)
        threshold1 = 1
        for i in range(0,10):
                crop.append(output[pin_crop_ranges[i][0]+y:pin_crop_ranges[i][1]+y1, pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
                hist = cv2.calcHist([crop[i]], [1], None, [4], [10, 50])
                sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
#                 print (i, sumHist[i])
                if threshold1 < sumHist[i]:
                    pinCount = pinCount + 2**(9-i)

        bit_GPIO(pinsGPIO,pinCount)
        if frameNo == 140:
             result = " _"+ str(priorPinCount)+"_" + str(pinCount) + "_" +str(frameNo)
             write_video(stream, result)
             return
        if pinsFalling == True:
                if timesup == False:
                    return
                else:
                    result = " _"+ str(priorPinCount)+"_" + str(pinCount) + "_" +str(frameNo)
                    print("FrameNo ", frameNo, "PinCount ", priorPinCount, "_",pinCount, result )
                    if priorPinCount == 1023 and priorPinCount != pinCount:  ## 1023 for full
                        write_video(stream, result)
                        if ballCounter == 0 and pinCount == 0:
                            flash()
                    priorPinCount = pinCount
                    pinsFalling = False
                    return
        if priorPinCount <= pinCount:
            priorPinCount = pinCount
            return
        else:
            pinsFalling = True
            t = threading.Timer(1.5, timeout)
            timesup = False
            t.start() # after 2.0 seconds, stream will be saved
            print('pin fallng timer is running', priorPinCount, pinCount)
            return

def iotSend(filename, result):
    global frameNo
    print("IOTSEND CALLED")

    client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
    try:
        print ("IoT Hub file upload sample, press Ctrl-C to exit")
        client.connect()
    

    # Get the storage info for the blob
        blob_name = os.path.basename(filename)
        vidfile = "videos/dp" + result + ".h264"
        storage_info = client.get_storage_info_for_blob(vidfile)
        print("blob_name2 +++: ", blob_name, filename, storage_info)

        # Upload to blob
        success, result = iot2.store_blob(storage_info, filename)

        if success == True:
            print("blob_name : ", blob_name)
            print("Upload succeeded. Result is: \n") 
            print(result)
            print()

            client.notify_blob_upload_status(
                storage_info["correlationId"], True, 200, "OK: {}".format(filename)
            )

        else :
            # If the upload was not successful, the result is the exception object
            print("Upload failed. Exception is: \n") 
            print(result)
            print()

            device_client.notify_blob_upload_status(
                storage_info["correlationId"], False, result.status_code, str(result)
            )
            blob_name = os.path.basename(filename)
            print("Blob_name ", blob_name)
    except KeyboardInterrupt:
        print ( "IoTHubClient sample stopped in iotSend" )
    finally:
    # Graceful exit
        client.shutdown()

def cleanupFinalFrameFiles():
    """
    Clean up any remaining local finalframe_pins_ files that may not have been deleted
    """
    try:
        video_dir = "/home/cliffeby/Videos"
        if os.path.exists(video_dir):
            for filename in os.listdir(video_dir):
                if filename.startswith("finalframe_pins_") and filename.endswith(".jpg"):
                    file_path = os.path.join(video_dir, filename)
                    try:
                        # Check if file is older than 1 hour to avoid deleting recently created files
                        file_age = time.time() - os.path.getmtime(file_path)
                        if file_age > 600000:  # about a week in seconds
                            os.remove(file_path)
                            print(f"Cleaned up old final frame file: {filename}")
                    except Exception as e:
                        print(f"Error cleaning up file {filename}: {e}")
    except Exception as e:
        print(f"Error during final frame cleanup: {e}")

def captureFinalFrame(final_frame, beginning_pin_count, ending_pin_count):
    """
    Captures the final frame of video, saves as JPG with pin count info, and uploads to Azure
    
    Args:
        final_frame: The final frame image from video
        beginning_pin_count: Pin count at start of ball roll
        ending_pin_count: Pin count at end of ball roll
    """
    global frameNo
    
    # Create unique identifier using timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    unique_id = str(int(time.time() * 1000))[-6:]  # Last 6 digits of millisecond timestamp
    
    # Create filename with pin counts and unique identifier
    local_filename = f"/home/cliffeby/Videos/finalframe_pins_{beginning_pin_count}_to_{ending_pin_count}_{timestamp}_{unique_id}.jpg"
    azure_filename = f"images/finalframe_pins_{beginning_pin_count}_to_{ending_pin_count}_{timestamp}_{unique_id}.jpg"
    
    # Save the frame as JPG
    success = cv2.imwrite(local_filename, final_frame)
    
    if success:
        print(f"Final frame saved locally: {local_filename}")
        
        # Upload to Azure
        client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
        try:
            print("Uploading final frame to Azure...")
            client.connect()
            
            # Get the storage info for the blob
            storage_info = client.get_storage_info_for_blob(azure_filename)
            
            # Upload to blob
            upload_success, upload_result = iot2.store_blob(storage_info, local_filename)
            
            if upload_success:
                print(f"Final frame uploaded successfully: {azure_filename}")
                print(f"Upload result: {upload_result}")
                
                client.notify_blob_upload_status(
                    storage_info["correlationId"], True, 200, "OK: {}".format(local_filename)
                )
                
                # Remove local file after successful upload
                try:
                    os.remove(local_filename)
                    print(f"Local file deleted: {local_filename}")
                except Exception as e:
                    print(f"Warning: Could not delete local file {local_filename}: {e}")
                
            else:
                print(f"Final frame upload failed: {upload_result}")
                client.notify_blob_upload_status(
                    storage_info["correlationId"], False, upload_result.status_code, str(upload_result)
                )
                
        except Exception as e:
            print(f"Error uploading final frame: {e}")
        finally:
            client.shutdown()
    else:
        print(f"Failed to save final frame locally: {local_filename}")

def iotSendImg(filename):
        global frameNo
        # v1 client.send_reported_state(reported_state, len(reported_state), iot.send_reported_state_callback, iot.SEND_REPORTED_STATE_CONTEXT)
        img_name = "images/imgdp" + time.strftime('%A') + ".jpg"
        f = open(filename, "rb+")
        content = f.read()
        
        print("CONTENT LEN", len(content), type(content))
        # v1 client.upload_blob_async(filename,content, len(content), iot.blob_upload_conf_callback,1001)

        # v2
        device_client = IoTHubDeviceClient.create_from_connection_string(CONNECTION_STRING)
        try:
            print ("IoT Hub file upload, press Ctrl-C to exit")
            device_client.connect()
        
        # Get the storage info for the blob
            blob_name = os.path.basename(filename)
            storage_info = device_client.get_storage_info_for_blob(img_name)

            # Upload to blob
            success, result = iot2.store_blob(storage_info, filename)

            if success == True:
                print("blob_name : ", blob_name)
                print("Upload succeeded. Result is: \n") 
                print(result)
                print()

                device_client.notify_blob_upload_status(
                    storage_info["correlationId"], True, 200, "OK: {}".format(filename)
                )

            else :
                # If the upload was not successful, the result is the exception object
                print("Upload failed. Exception is: \n") 
                print(result)
                print()

                device_client.notify_blob_upload_status(
                    storage_info["correlationId"], False, result.status_code, str(result)
                )
#                 blob_name = os.path.basename(filename)
                print("Blob_name ", blob_name)
        except KeyboardInterrupt:
            print ( "IoTHubClient sample stopped" )
        finally:
        # Graceful exit
            device_client.shutdown()
# 
#     iot.print_last_message_time(client)

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
    cv2.imwrite('/home/cliffeby/Videos/CCEPinMask'+str(i) +'.jpg',ball_image)
    iotSendImg('/home/cliffeby/Videos/CCEPinMask9.jpg')
    
setupGPIO(pinsGPIO)
setupGPIO(segment7s)
tripSet()
priorPinCount = 0
pinsFalling = False
timesup = True
activity = "\r\n"
x = offsets.offset['x']  #(minus) moves blue crops x pixels left
x1 = 0 +x
y = offsets.offset['y']  # -(minus) moves blue crops y pixels up1
y1 = 0 + y
frameNo = 0
ballCounter = 0
videoReadyFrameNo = 10
timesupDeadwood = True
timesupReset = True
# deadwoodTimer = time.time()
lastCleanupTime = time.time()  # Track when we last cleaned up files
lightsOFF(segment7s)
# Kepp ball counter off
# GPIO.output((segment7All[0]), GPIO.LOW)

with picamera.PiCamera() as camera:
    camera.resolution = setResolution()
    camera.video_stabilization = True
    camera.annotate_background = True
    camera.rotation = 180
    rawCapture = PiRGBArray(camera, size=camera.resolution)
    # setup a circular buffer
    # stream = picamera.PiCameraCircularIO(camera, seconds = video_preseconds, size = bytes)
    stream = picamera.PiCameraCircularIO(camera, size = 4000000)
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
            #  Ball counter eliminated
            # if GPIO.input(sensor[0]) == 0 and timesupReset == True:
            #     ballCounter= ballCounter+1
            #     print ("Ball Falling edge", ballCounter)
            #     lightsOFF(segment7s)
            #     GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
            #     print('Ball Timer Awake ', ballCounter)               
        if (GPIO.input(sensor[1]) == GPIO.HIGH):
                print('Deadwood sensor', ballCounter)
                if timesupDeadwood == True:
                    tDeadwood = threading.Timer(12.0, timeoutDeadwood)
                    timesupDeadwood = False
                    tDeadwood.start()
                    print ('Deadwood timer has started', ballCounter)
        if (GPIO.input(sensor[2]) == GPIO.HIGH):
                print('Reset sensor-pre', ballCounter)
                ballCounter = 0
                lightsOFF(segment7s)
                GPIO.output((segment7All[0]), GPIO.LOW)
                bit_GPIO(pinsGPIO,1023)
                GPIO.wait_for_edge(sensor[2], GPIO.FALLING)
                if timesupReset == True:
                    tReset = threading.Timer(5.0, timeoutReset)
                    timesupReset = False
                    tReset.start()
                    print ('Reset timer is running', ballCounter)

        writeImageSeries(30, 1, img_rgb)
        # if deadwoodTimer+10<time.time():
        #     print(deadwoodTimer, time.time())
        
        # Periodic cleanup of old final frame files (every 10 minutes)
        if time.time() - lastCleanupTime > 600:  # 600 seconds = 10 minutes
            cleanupFinalFrameFiles()
            lastCleanupTime = time.time()
            
        if frameNo%4== 0:
            print(timesup,timesupDeadwood,timesupReset, frameNo, GPIO.input(sensor[0]),GPIO.input(sensor[1]),GPIO.input(sensor[2]))
            if timesupDeadwood and timesupReset:
                findPins()
        # else:
        #     print('Skipped findPins()')
