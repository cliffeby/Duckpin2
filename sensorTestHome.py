# import the necessary packages


import RPi.GPIO as GPIO
import myGPIO
import time

sensor = myGPIO.sensor
# sensor[0] = 17
ballCounter = 0


def setupGPIO(pins):
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
    print("setup Completed")


def tripSet():
    global sensor
    for s in sensor:
        GPIO.setup(s, GPIO.OUT)
        GPIO.output(s, GPIO.LOW)
        time.sleep(.5)
        GPIO.setup(s, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    print('Sensor setup complete')


setupGPIO(sensor)
tripSet()
done = True

while True:

    # print('Sensor', ballCounter, GPIO.input(sensor[0]), GPIO.input(
    #     sensor[1]), GPIO.input(sensor[2]))

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
            print('done')
            done = True
            time.sleep(.05)
            if GPIO.input(sensor[0]) == 0:

                ballCounter = ballCounter+1
                print("FALLING", ballCounter)
        except KeyboardInterrupt:
            GPIO.cleanup()
    # if done == True:
    #     done = False
    #     ballCounter = ballCounter + 1
    #     # lightsOFF(segment7s)
    #     # GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
    #     print('Ball Timer Awake ', ballCounter)

    while (GPIO.input(sensor[1]) == GPIO.HIGH):
            # GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
        print('Deadwood ', ballCounter)

    while (GPIO.input(sensor[2]) == GPIO.HIGH):
            # GPIO.output((segment7All[ballCounter % 10]), GPIO.LOW)
        ballCounter = 0
        print('Reset ', ballCounter)
