import cv2
import numpy as np
import cropdata1440
from matplotlib import pyplot as plt


def getCroppedImage(image, crop_array):
    croppedImage = image[crop_array[0]
        :crop_array[1], crop_array[2]:crop_array[3]]
    return croppedImage


def findPins(img):
    global x, x1, y, y1
    global priorPinCount, frameNo
    global pinsGPIO
    global frame2
    global pinsFalling, timesup  # initial values False, True
    global upper_red, lower_red
    pinCount = 0
    crop = []
    sumHist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     lower_red = np.array([0, 0, 70])  # lower_red = np.array([0,100,0])
#     # upper_red = np.array([180,255,255])
#     upper_red = np.array([110, 110, 255])
#     lower_red = np.array([170, 100, 100])  # lower_red = np.array([0,100,0])
#     # upper_red = np.array([180,255,255])
#     upper_red = np.array([180, 255, 255])

    mask = cv2.inRange(img, lower_red, upper_red)
    output = cv2.bitwise_and(img, img, mask=mask)
    threshold1 = 10
    for i in range(0, 10):
        crop.append(output[pin_crop_ranges[i][0]+y:pin_crop_ranges[i]
                           [1]+y1, pin_crop_ranges[i][2]+x:pin_crop_ranges[i][3]+x1])
        hist = cv2.calcHist([crop[i]], [0], None, [4], [150, 200])
        sumHist[i] = hist[0]+hist[1]+hist[2]+hist[3]
        print(i,  sumHist[i])
        # plt.hist(crop[i].ravel(), 256, [0, 256])
        # plt.show()


x = y = x1 = y1 = 0
pin_crop_ranges = cropdata1440.pin_crop_ranges
img = cv2.imread("/home/cliffeby/Downloads/imgdpThursday (1).jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('ddd', img)

# ret, thresh_basic = cv2.threshold(img, 50, 200, cv2.THRESH_BINARY)
# cv2.imshow('tt', thresh_basic)

# thres_adapt = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
# cv2.imshow('aa', thres_adapt)
# minimum value of blue pixel in BGR order
lower_red = np.array([170, 100, 100])  # lower_red = np.array([0,100,0])
# upper_red = np.array([180,255,255])
upper_red = np.array([180, 255, 255])
# lower_red = np.array([160, 100, 100])  # upper_red = np.array([110, 110, 255])
# upper_red = np.array([179, 255, 255])
# img = cv2.inRange(img, lower_red, upper_red)
# cv2.imshow('r', img)
# no_blue = cv2.countNonZero(dst)
# print('The number of blue pixels is: ' + str(no_blue))
# cv2.namedWindow("opencv")
# cv2.imshow("opencv", img)
findPins(img)
# img2 = getCroppedImage(img, pin_crop_ranges[9])
# plt.hist(img2.ravel(), 256, [0, 256])
# plt.show()

# color = ('b', 'g', 'r')
# for i, col in enumerate(color):
#     histr = cv2.calcHist([img2], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#     plt.xlim([0, 256])
# plt.show()

img1 = cv2.imread("/home/cliffeby/Downloads/imgdpThursday (1).jpg")
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
cv2.imshow('ddd1', img1)

# img3 = getCroppedImage(img1, pin_crop_ranges[9])
# for i, col in enumerate(color):
#     histr = cv2.calcHist([img3], [i], None, [256], [0, 256])
#     plt.plot(histr, color=col)
#     plt.xlim([0, 256])
# plt.show()

# ret, thresh_basic1 = cv2.threshold(img1, 50, 200, cv2.THRESH_BINARY)
# cv2.imshow('tt1', thresh_basic1)
# lower_red = np.array([0, 0, 70])  # lower_red = np.array(  [0, 0, 70])
# upper_red = np.array([50, 50, 190])  # upper_red = np.array([110, 110, 255])
# dst = cv2.inRange(img1, lower_red, upper_red)
# cv2.imshow('r1', dst)
findPins(img1)

cv2.waitKey(0)
cv2.destroyAllWindows
