import numpy as np
import cv2
import cropdata1440

pin_crop_ranges = cropdata1440.pin_crop_ranges

def getCroppedImage(image,crop_array):
    croppedImage = image[crop_array[0]:crop_array[1],crop_array[2]:crop_array[3]]
    return croppedImage

def findPins(img):
        global x,x1,y,y1
        global priorPinCount, frameNo, ballCounter
        global black_img
        global frame2
        global pinsFalling, timesup
        crop = []# initial values False, True

        for i in range(0,10):
                crop.append(img[pin_crop_ranges[i][0]+y:pin_crop_ranges[i][1]+y1, pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
                avg_color_per_row = np.average(crop[i], axis=0)
                avg_color = np.average(avg_color_per_row, axis = 0)
                a =crop[i].mean()
                print (i, pin_crop_ranges[i][0]+y, avg_color, a)
        return

x=35   # - (minus) moves crops x pixels right
x1=0 +x
y=-55  # -(minus) moves crop y pixels up
y1=0 + y
image = cv2.imread('/home/cliffeby/Downloads/imgdpThursday (2).jpg')
result = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
black_img = image.copy()# Set using redslide on a 08/19/22 image
findPins(black_img)

cv2.imshow('result', result)
cv2.waitKey()