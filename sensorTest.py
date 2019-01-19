# import the necessary packages


import RPi.GPIO as GPIO
import time

sensor = [37, 40, 38]
# sensor = [26, 21, 20]
ballCounter = 0


def setupGPIO(pins):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
    print("setup Completed")


def tripSet():
    global sensor, ballCounter, segment7s, segment7All
    for s in sensor:
        GPIO.setup(s, GPIO.OUT)
        GPIO.output(s, GPIO.LOW)
        time.sleep(.5)
        GPIO.setup(s, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)


setupGPIO(sensor)
tripSet()
done = True

while True:

    print('Sensor', ballCounter, GPIO.input(sensor[0]), GPIO.input(
        sensor[1]), GPIO.input(sensor[2]))

    # while (GPIO.input(sensor[0]) == GPIO.HIGH):
    #     ballCounter = ballCounter + 1
    #     # GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
    #     print('Ball ', ballCounter)
    while (GPIO.input(sensor[0]) == GPIO.HIGH):
                # GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
                # print('Ball Timer Awake ', ballCounter)
        done = False
        try:
            GPIO.wait_for_edge(sensor[0], GPIO.FALLING)
            print('Falling edge')
            done = True
        except KeyboardInterrupt:
            GPIO.cleanup()
    if done == True:
        done = False
        ballCounter = ballCounter + 1
        # lightsOFF(segment7s)
        # GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
        print('Ball Timer Awake ', ballCounter)

    while (GPIO.input(sensor[1]) == GPIO.HIGH):
            # GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
        print('Deadwood ', ballCounter)

    while (GPIO.input(sensor[2]) == GPIO.HIGH):
            # GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
        ballCounter = 0
        print('Reset ', ballCounter)
