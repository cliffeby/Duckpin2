from picamera import PiCamera, Color
from time import sleep
import datetime as dt
import time

camera = PiCamera()
camera.resolution = (640, 480)


def camPreview(howLong):
    camera.start_preview()
    sleep(howLong)
    camera.stop_preview()


def camRotation(degrees):
    camera.start_preview
    camera.rotation = 90
    camera.start_preview()
    sleep(1)
    camera.stop_preview()


def camBrightness():
    camera.start_preview()
    camera.annotate_background = Color('blue')
    camera.annotate_foreground = Color('yellow')
    for i in range(100):
        camera.brightness = i
        camera.annotate_text = "Brightness = " + str(i)
        sleep(0.15)
    camera.stop_preview()
    camera.brightness = 50
    camera.annotate_background
    camera.annotate_foreground


def camContrast():
    camera.start_preview()
    camera.annotate_background = Color('blue')
    camera.annotate_foreground = Color('yellow')
    for i in range(-100, 100):
        camera.contrast = i
        camera.annotate_text = "Contrast = " + str(i)
        sleep(0.15)
    camera.stop_preview()
    camera.contrast = 0
    camera.annotate_background
    camera.annotate_foreground


def camPicCapture(numberOfPics, delay):
    camera.start_preview()
    camera.annotate_background = Color('blue')
    camera.annotate_foreground = Color('yellow')
    camera.annotate_text = " Hello duckpin world " + dt.datetime.now().isoformat('0')
    for i in range(numberOfPics):
        sleep(delay)
        camera.annotate_text = " Hello duckpin1 world " + dt.datetime.now().isoformat('0')
        camera.capture('/home/pi/Shared/images/image%s.jpg' % i)
        print('Pics saved ', i)
    camera.stop_preview()
    camera.annotate_text = ""
    camera.annotate_background
    camera.annotate_foreground


def camVideoCapture(numberOfVideos, videoLength, delayBetweenVideos):
    camera.start_preview()
    camera.annotate_background = Color('blue')
    camera.annotate_foreground = Color('yellow')
    for i in range(numberOfVideos):
        sleep(delayBetweenVideos)
        camera.annotate_text = " Hello duckpin1 world " + dt.datetime.now().isoformat('0')
        camera.start_recording('/home/pi/Shared/videos/video%s.h264' % i)
        sleep(videoLength)
        camera.stop_recording()
        print('Video saved ', i)
    camera.stop_preview()
    camera.annotate_text = ""
    camera.annotate_background
    camera.annotate_foreground


# camera.vflip = True
# camera.hflip = False
# camRotation(0)
# camPreview(10)
# camBrightness()
# camContrast()
camPicCapture(5, 1)
camVideoCapture(3, 3, 1)
