# import the necessary packages
import random, string
import datetime
import credentials
from azure.storage import CloudStorageAccount
from azure.storage.table import TableService, Entity
from azure.storage.blob import BlockBlobService, PageBlobService, AppendBlobService
import cropdata1440
import time
import cv2
import numpy
import glob
import os, sys

account_name = credentials.STORAGE_ACCOUNT_NAME
account_key = credentials.STORAGE_ACCOUNT_KEY
account = CloudStorageAccount(account_name, account_key)
ball_crops = cropdata1440.ballCrops

def basic_blockblob_operations(account):

        # # Create a Block Blob Service object
        blockblob_service = account.create_block_blob_service()
        blockblob_service = BlockBlobService(account_name, account_key)
        
        # List all the blobs in the container 
        dpContainer = 'jsoncontdp'
        print('List Blobs in Container')
            
        generator = blockblob_service.list_blobs(dpContainer)
        if sum(1 for _ in generator) == 0:
            print('No blobs to process -- program ending')
            exit(0)

        for blob in generator:
            print('\tBlob Name: ' + blob.name)
            file = blob.name
        # Download the blob
            downloadDir = 'c:/DownloadsDP/' 
            print('Download the blob', file, 'path', downloadDir)
        # blockblob_service.get_blob_to_path(dpContainer, file, os.path.join(os.path.dirname(__file__)+'/'+ file))
            blockblob_service.get_blob_to_path(dpContainer, file, downloadDir+file)
        # blockblob_service.get_blob_to_path(container_name, file_to_upload, os.path.join(os.path.dirname(__file__), file_to_upload + '.copy.png'))
        
        # Clean up after the sample
            print('Delete  Blob ', downloadDir+file)
            blockblob_service.delete_blob(dpContainer, file)
        
        # # Delete the container
        # print("6. Delete Container")
        # blockblob_service.delete_container(container_name)
        

def findBeg(file):
    index = 0
    loc = []
    pinsBA = [0,0]
    while index < len(file):
        index = file.find('_', index)
        if index==-1:
            # No more underscores found in string
            break
        loc.append(index)
        index = index+1 
    pinsBA[0] = int(file[loc[0]+1:loc[1]])
    pinsBA[1] = int(file[loc[1]+1:loc[2]])
    return pinsBA

def formatxy(pinData):
    xy = {}
    counter = 0
    while counter < len(pinData and counter < 6):
            extra = {'x' + str(counter) : str(pinData[counter][0]) , 'y' +str(counter): str(pinData[counter][1])}
            xy.update(extra)
            counter = counter+1
    # print ('XY DICT', xy)
    return xy

def insertRows(file, xy):
        global pinData, account
        print('Azure Storage Table Duckpins - Starting row entry.')
        table_service = None
           
        def getRowKey():
            x= datetime.datetime.now()
            rowID = x.strftime('%Y')+x.strftime('%m')+ x.strftime('%d')+x.strftime('%f')
            return rowID
        
        table_service = account.create_table_service()
        table_name = 'pindata'
        # Create a new table
        try:
            table_service.create_table(table_name)
        except Exception as err:
            print('Error creating table, ' + table_name + 'check if it already exists')
        pinevent ={'PartitionKey':'Lane 4','RowKey': str(getRowKey()), 'beginingPinCount': findBeg(file)[0], 'endingPinCount': findBeg(file)[1] }
        pinevent.update(xy)
       # Insert the entity into the table
        table_service.insert_entity(table_name, pinevent)        
        print('Successfully inserted the new entityinto table - ' + table_name, pinevent)



def getCroppedImage(image,crop_array):
    croppedImage = image[crop_array[0]:crop_array[1],crop_array[2]:crop_array[3]]
    return croppedImage

def isPinSetter():
    global setterPresent
    global frameNo
    global img_rgb
    global firstSetterFrame  
    # Convert BGR to HSV
    frame = img_rgb[150:450, 650:1400]
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

def cleanup():
    a = glob.glob('C:/DownloadsDP/Lane4Free/dp*.h264')
    fileCounter = 0
    while fileCounter < len(a):
        if os.path.isfile(a[fileCounter]):
            # os.remove(a[fileCounter])
            print('Deleted file', fileCounter, a[fileCounter])
        else:    ## Show an error ##
            print("Error: %s file not found" % a[fileCounter])
        fileCounter += 1


basic_blockblob_operations(account)
a = []
b = {}
a = glob.glob('C:/DownloadsDP/Lane4Free/dp*.h264')
xyData = [0,0]
pinData = []   
cap = cv2.VideoCapture(a[0])
# setupGPIO(pinsGPIO)
setterPresent = False
armPresent = False
priorPinCount = 0
x=0
x1=0 +x
y=-0
y1=0 + y
fileCounter = 1
frameNo = 0
prevFrame = 0
ret, frame1 = cap.read()
frame1= getCroppedImage(frame1,ball_crops)
img_gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
img_gray_show = img_gray1.copy()
# getCroppedImage(frame1, mask_crop_ranges)

while(cap.isOpened()):
    ret, frame2 = cap.read()
    try:
        type(frame2[0]) is None
    except:
        print ("End of Video ", fileCounter)
        
        if fileCounter < len(a):
            cap.release()
            xy = formatxy(pinData)
            if len(xy)>0:
                insertRows(a[fileCounter], xy)
            else:
                print('No ball data in Video ', fileCounter)
            cap = cv2.VideoCapture(a[fileCounter])
            ret, frame2 = cap.read()
            fileCounter = fileCounter+1
            pinData = []
        else:
            print('No more data to process')
            cap.release()
            cleanup()
            break
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

    # isPinSetter()
    frame2= getCroppedImage(frame2,ball_crops)
    
    img_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(img_gray1,img_gray2)
    # First value reduces noise.  Values above 150 seem to miss certain ball colors
    ret, thresh = cv2.threshold(diff, 120,255,cv2.THRESH_BINARY)
    # frame = thresh
    # Blur eliminates noise by averaging surrounding pixels.  Value is array size of blur and MUST BE ODD
    thresh = cv2.medianBlur(thresh,5)
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

		# only proceed if the radius meets a minimum size
        if radius > 20:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cv2.drawContours(img_gray2, cnts, -1, (0,255,0), 3)
            # if center < (1100,200):
            # print('CENTER',center, radius, frameNo, len(cnts))
            xyData = (center[0], center[1])
            pinData.append(xyData)
            cv2.drawContours(img_gray_show, cnts, -1, (0,255,0), 3)

    # cv2.imshow('Ball' , img_gray2)
    cv2.imshow('All' , img_gray_show)
    # tf = findPins()

    # cv2.rectangle(img_rgb,b, a, 255,2)
    
    key = cv2.waitKey(1) & 0xFF
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
