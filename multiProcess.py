from multiprocessing import Process, Value, Array, Queue, Pipe
import time
from datetime import datetime
import cv2
import numpy
import RPi.GPIO as GPIO
import Iothub_client_functions as iot
import picamera
from picamera.array import PiRGBArray
import picamera.array


def f(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]
    time.sleep(2)
    print(a[:])


def trip(conn):
    ballCounter = 0

    while True:
        ballCounter = ballCounter + 1
        # time.sleep(.001)
        #     print('Ball Timer Awake ', ballCounter)
        timesup = True
        # temp = q.get()
        if ballCounter % 100000 == 0:
            x = conn.recv()
            conn.send(ballCounter)
            print('Multi', ballCounter, type(x), x.size, x.shape)


if __name__ == '__main__':
    num = Value('d', 0.0)
    arr = Array('i', range(10))

    q = Queue()
    trip_conn, other_conn = Pipe()

    p = Process(target=f, args=(num, arr))
    p.start()
    p.join()
    #     getRGB(q)
    #     p1 = Process(target=getRGB, args=(q,))
    #     p1.start()
    #     while True:
    #         if q.get() % 10 == 0:

    #             print('FrameNo', q.get())

    #     p1.join()
    #     getRGB()
    p2 = Process(target=trip, args=(other_conn,))
    p2.start()

    print(num.value)
    print(arr[:])
    frameNo = 0
    with picamera.PiCamera() as camera:
        # camera.resolution = setResolution()
        camera.video_stabilization = True
        camera.annotate_background = True
        camera.rotation = 180
        rawCapture = PiRGBArray(camera)
        # setup a circular buffer
        # stream = picamera.PiCameraCircularIO(camera, seconds = video_preseconds)
        stream = picamera.PiCameraCircularIO(camera, size=3000000)
        # video recording into circular buffer from splitter port 1
        camera.start_recording(stream, format='h264', splitter_port=1)
        #camera.start_recording('test.h264', splitter_port=1)
        # wait 2 seconds for stable video data
        camera.wait_recording(2, splitter_port=1)
        # motion_detected = False
        print(camera.resolution)
        startTime = time.time()
        print('Preframe')

        for frame in camera.capture_continuous(rawCapture, format="bgr",  use_video_port=True):
            # grab the raw NumPy array representing the image, then initialize the timestamp
            # and occupied/unoccupied text???????????????????
            rawCapture.truncate()
            rawCapture.seek(0)

            frame2 = frame.array
            frameNo = frameNo + 1

            trip_conn.send(frame2)
            print(frameNo)
            print('CONN', trip_conn.recv())
            cv2.imshow('RGB', frame2)

            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                camera.close()
                break
