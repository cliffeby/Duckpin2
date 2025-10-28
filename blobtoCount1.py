# import the necessary packages
import copy
import datetime
import glob
import math
import os
import time
import credentials
import cv2
import numpy as np
import cropdata1440  # defines ball crops - area before ball hits pins
import shutil
from azure.storage.blob import BlobServiceClient
from azure.core.credentials import AzureNamedKeyCredential
from azure.data.tables import TableServiceClient
from azure.core.credentials import AzureNamedKeyCredential
from azure.identity import DefaultAzureCredential

account_url = "https://duckpinjson.blob.core.windows.net"
account_urlTable = "https://duckpinjson.table.core.windows.net"
default_credential = DefaultAzureCredential()
account_name = credentials.STORAGE_ACCOUNT_NAME
account_key = credentials.STORAGE_ACCOUNT_KEY
account = AzureNamedKeyCredential(account_name, account_key)
table_name='pindata'
container = 'jsoncontdp'
downloadDir = 'c:/DownloadsDP/' 
ball_crops = cropdata1440.ballCrops
ball_crops = [460, 885, 10, 1200]

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient(account_url, credential=account)
blob_client = blob_service_client.get_blob_client(container=container, blob='any_file_name')
container_client = blob_service_client.get_container_client(container)
print("\nListing blobs...")

# List the blobs in the container
blob_list = container_client.list_blobs()
for blob in blob_list:
    print("\t" + blob.name)
    file = blob.name
    
# Download the blob
    container_client.get_blob_client(container,file)
    
    file1 = downloadDir+file

    with open(file1, "wb") as my_blob:
        download_stream = container_client.download_blob(file)
        my_blob.write(download_stream.readall())

        if ".h264" in file:
            print('Delete  Blob ', downloadDir+file)
            container_client.delete_blob(file)

def findBeg(file):
    # Parse file name for beginning and ending pin state
    # File name format is C:/DownloadsDP/Lane4Free\dp _1023_695_.h264
    index = 0
    loc = []
    pinsBA = [0,0]
    while index < len(file):
        index = file.find('_', index)
        if index == -1:
            # No more underscores found in string
            break
        loc.append(index)
        index = index+1 
    pinsBA[0] = int(file[loc[0]+1:loc[1]])
    pinsBA[1] = int(file[loc[1]+1:loc[2]])
    return pinsBA

def formatxy(pinData):
    # Put ball xy data in json format
    xy = {} # create empty dictionary
    counter = 0
    while counter < len(pinData) and counter < 6:  # counter > 5 - to elinimate lingering ball or downed pin noise
            extra = {'x' + str(counter) : str(pinData[counter][0]) , 'y' +str(counter): str(pinData[counter][1])}
            xy.update(extra)
            counter = counter+1
    return xy

def insertRows(file, xy):
    # Insert row into Azure table
    global pinData, account
    print('Azure Storage Table Duckpins - Starting row entry.')
    # table_service = None
        
    def getRowKey():
        #Azure tables need a unique row key
        x= datetime.datetime.now()
        rowID = x.strftime('%Y')+x.strftime('%m')+ x.strftime('%d')+x.strftime('%f')
        return rowID

    table_service_client = TableServiceClient(account_urlTable, credential=account)
    table_client = table_service_client.get_table_client(table_name=table_name)

    # Create a table in case it does not already exist
    # try:
    #     table_client.create_table()
    # except Exception as err:
    #     print('Error creating table, '  + 'check if it already exists', err)
    rowkey = str(getRowKey())
    pinevent = {'PartitionKey':'Lane 4','RowKey': rowkey,'res':'1440', 'beginingPinCount': findBeg(file)[0], 'endingPinCount': findBeg(file)[1] }
    if len(xy) < 2:  #In a dictionary the key and value are counted as one entry
        print('Entry only contains one xy pair and has been removed ', rowkey)
        return
    if findBeg(file)[1] == 1023:  # Eliminate gutter or event with no pin action
        return
    pinevent.update(xy)
    # Insert the entity into the table
    table_client.create_entity( pinevent)        
    print('Successfully inserted the new entity into table - ' +str(not colorFlag)+ ' ' + file, table_name, pinevent)
    with open("C:/DownloadsDP/Lane4Free/rows.txt", "a") as myfile:
        row = str( not colorFlag) + ' '+ rowkey+ " " + str(findBeg(file)[0]) + ' ' +str(findBeg(file)[1]) + str(xy) + '\n'
        myfile.write(row)

def dist(old, new, thresh):
    # Checks for very slow ball movement or arm looking like a ball
    # Is ball moving greater than trhesh in pixels
    l2 = (old[0] - new[0])**2 + (old[1] -new[1])**2
    if math.sqrt(l2) < thresh:
        return True
    # Is ball moving backward - y axis
    if old[1] - new[1] < 0:
        return True
    # Is ball moving sideways (x-axis) or is sensor #2 not detecting deadwood or reset
    # if abs(old[0] - new[0] +10) > 20:
    #     print('Arm detected')
    #     return True
    return False

def getCroppedImage(image, crop_array):
    croppedImage = image[crop_array[0]:crop_array[1],crop_array[2]:crop_array[3]]
    return croppedImage

def cleanup():
    a = glob.glob('C:/DownloadsDP/Lane4Free/dp*.h264')
    fileCounter = 0
    # directory = 'C:/DownloadsDP/TempHold/'
    while fileCounter < len(a):
        if os.path.isfile(a[fileCounter]):
            # shutil.move(a[fileCounter],directory)
            # print('Moved file', fileCounter)
            # file.close()
            os.remove(a[fileCounter])
            print('Deleted file', fileCounter, a[fileCounter])
        else:    ## Show an error ##
            print("Error: %s file not found" % a[fileCounter])
        fileCounter += 1

def my_division(n, d):
    return n / d if d else 0

def fixskew(img):
    pts1 = np.float32([[240,550],[1050,580],[0,1000],[1295,1000]]) # skewed area in front of pins
    pts2 = np.float32([[0,0],[875,0],[0,460],[1000,460]]) # rectangular area
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(1000,400))
    return dst

# basic_blockblob_operations(account)
a = []
a = glob.glob('C:/DownloadsDP/Lane4Free/dp*.h264')
xyData = [0,0]
oldxyData = None
pinData = []
fileCounter = 0
cap = cv2.VideoCapture(a[1])
ret, frame0 = cap.read()
frame1= cv2.imread('C:/DownloadsDP/Lane4Free/dplane.jpg')
fxImage = fixskew(frame1)
# frame1 = getCroppedImage(frame1, ball_crops)
frame1 = fixskew(frame1)
img_gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
# fxImage = fixskew(img_gray1)
cv2.imwrite('C:/DownloadsDP/TempHold1/dpballfxdskew'+ time.strftime("%Y%m%d") +'.jpg', fxImage)

# cv2.imwrite('C:/DownloadsDP/Lane4Free/dplane.jpg',frame0)
# frame1= cv2.imread('C:/DownloadsDP/Lane4Free/dplane.jpg')
# frame1 = getCroppedImage(frame1, ball_crops)
# img_gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
img_gray_show = copy.deepcopy(img_gray1)
img_gray_show_line = copy.deepcopy(img_gray1)
colorFlag = False
while (cap.isOpened()):
    ret, frame2 = cap.read()
    try:
        type(frame2[0]) is None
    except:
        print("End of Video ", fileCounter)
        
        if fileCounter < len(a)-1:
            # Save the final frame before moving to next video
            if 'last_valid_frame' in locals():
                final_frame_path = f'C:/DownloadsDP/Lane4Free/final_frame_video_{fileCounter}_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
                cv2.imwrite(final_frame_path, last_valid_frame)
                print(f'Saved final frame: {final_frame_path}')
            
            cap.release()
            xy = formatxy(pinData)
            if len(xy) > 0:
                insertRows(a[fileCounter], xy)
            else:
                print('No ball data in Video ', fileCounter)
            # Get new video file
            fileCounter = fileCounter+1
            oldxyData = None
            cap = cv2.VideoCapture(a[fileCounter])
            ret, frame2 = cap.read() 
            pinData = []
            img_gray_show = copy.deepcopy(img_gray1)
            colorFlag = False
        else:
            # Save the final frame of the last video
            if 'last_valid_frame' in locals():
                final_frame_path = f'C:/DownloadsDP/Lane4Free/final_frame_video_{fileCounter}_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
                cv2.imwrite(final_frame_path, last_valid_frame)
                print(f'Saved final frame of last video: {final_frame_path}')
            
            cap.release()
            xy = formatxy(pinData)
            if len(xy)>0:
                insertRows(a[fileCounter], xy)
            else:
                print('No ball data in final Video ', fileCounter)
            print('No more data to process')
            cv2.imwrite('C:/DownloadsDP/Lane4Free/dpballgrayline'+ time.strftime("%Y%m%d") +'.jpg', img_gray_show_line)
            print('Saving line image ')
            cleanup()  # Delete files from local storage
            break
    img_rgb = frame2
    if frame2 is None:
        continue
    
    # Store this frame as the last valid frame
    last_valid_frame = frame2.copy()
    frame2 = fixskew(frame2)
    img_gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(img_gray1, img_gray2)
    # First value reduces noise.  Values above 150 seem to miss certain ball colors
    ret, thresh = cv2.threshold(diff, 80, 255, cv2.THRESH_BINARY)
    # Blur eliminates noise by averaging surrounding pixels.  Value is array size of blur and MUST BE ODD
    thresh = cv2.medianBlur(thresh, 5)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    radius = 0
    if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((xContour, yContour), radius) = cv2.minEnclosingCircle(c)
		# only proceed if the radius meets a minimum size
        if radius > 30:
            # Find the xy center of the ball in the frame then,
            # draw the circle on the frame,
            # then update the list of tracked points
            M = cv2.moments(c)
            center = (int(my_division(M["m10"],M["m00"])), int(my_division(M["m01"], M["m00"])))
            xyData = (center[0], center[1])
            pinData.append(xyData)
            cv2.drawContours(img_gray_show, c, -1, (0, 255, 0), 3)
            # Eliminate centers of early half ball contours
            if center[1]>380:
                xyData = oldxyData
                continue
            if xyData[0]> 976:
                xyData = oldxyData
                continue
            # Eliminate centers of slow and backward moving balls
            elif oldxyData != None:
                if dist(oldxyData, xyData, 5):
                    pinData.pop()
                    xyData = oldxyData
                    print('Ball not moving - dist and location', xyData, radius)
                else:
                    if abs(oldxyData[0]-xyData[0])>20 or colorFlag:
                        cv2.line(img_gray_show_line, (oldxyData[0], oldxyData[1]),(xyData[0], xyData[1]), (255, 0, 0), 1)
                        colorFlag = True
                        print('Flagged',oldxyData,xyData, 'File', fileCounter)
                        # font
                        font = cv2.FONT_HERSHEY_SIMPLEX

                        # org
                        org = (oldxyData[0], oldxyData[1])

                        # fontScale
                        fontScale = 1
                        
                        # Blue color in BGR
                        color = (255, 0, 0)

                        # Line thickness of 2 px
                        thickness = 1
                        cv2.putText(img_gray_show_line, str(fileCounter), org, font, fontScale, color, thickness)
                    else:
                        cv2.line(img_gray_show_line, (oldxyData[0], oldxyData[1]),(xyData[0], xyData[1]), (0, 255, 0), 1)
                    cv2.circle(img_gray_show_line, xyData, 3, (0, 255, 0), -1)
                    cv2.circle(img_gray_show_line, oldxyData, 3, (0, 255, 0), -1)
                
            oldxyData = xyData

    cv2.imshow('Ball locations' , img_gray_show)
    cv2.imshow('Ball line' , img_gray_show_line)
    cv2.imshow('Frame2' , frame2)
    # if frameNo < 100:
    #     cv2.imwrite('C:/DownloadsDP/Lane4Free/dpballgray' +str(frameNo) +'.jpg',img_gray_show )
    #     print('Saving image ', frameNo)
    
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
