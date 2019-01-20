import RPi.GPIO as GPIO
import time
import myGPIO
from random import randint


leds = myGPIO.segment7s#[8,24,23,15,7,25,14]
led0 = myGPIO.segment7_0#[8,24,23,15,7,25]
led1 = myGPIO.segment7_1#[8,24]
led2 = myGPIO.segment7_2#[8,23,15,25,14]
led3 = myGPIO.segment7_3#[8,24,23,25,14]
led4 = myGPIO.segment7_4#[8,24,7,14]
led5 = myGPIO.segment7_5#[24,23,7,25,14]
led6 = myGPIO.segment7_6#[24,23,15,7,14]
led7 = myGPIO.segment7_7#[8,24,25]
led8 = myGPIO.segment7_8#[8,24,23,15,7,25,14]
led9 = myGPIO.segment7_9#[8,24,23,7,25,14]
ledAll = [led0, led1, led2, led3, led4, led5, led6, led7, led8, led9]


def setupGPIO(pins):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    for pin in pins:
        GPIO.setup(pin, GPIO.OUT)
        print("setup Completed")


def lightsOFF(pins):
    for pin in pins:
        GPIO.output(pin, GPIO.HIGH)
    print("ALL off")

# Turn on each light for x seconds


def lightTESTa(pins, wait1, wait2):
    i=0
    for pin in pins:
        print("Pin ", pin, i ,"is on")
        GPIO.output(pin, GPIO.LOW)
        time.sleep(wait1)
        print("Pin ", pin, "is off")
        i=i+1
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(wait2)


def numTEST(pins, wait1):
    for pin in pins:
        GPIO.output(pin, GPIO.LOW)
    time.sleep(wait1)


def listTEST(pins):
    for pin in pins:
        lightsOFF(leds)
        numTEST(pin, 1)


def timeout():
    global timesup
    timesup = True
    print('Timer is finished', timesup)

wait1 = 2

setupGPIO(leds)
lightsOFF(leds)

lightTESTa(leds,1.1,0)
lightTESTa(leds[::1], 10.5,0)
lightsOFF(leds)
# listTEST(ledAll)
# trip()
# lightTESTa(leds,.1,0)
# numTEST(led0,1)
# lightsOFF(leds)
# numTEST(led1,1)
# lightsOFF(leds)
# numTEST(led2,1)
# lightsOFF(leds)
# numTEST(led3,1)
# lightsOFF(leds)
# numTEST(led4,1)
# lightsOFF(leds)
# numTEST(led5,1)
# lightsOFF(leds)
# numTEST(led6,1)
# lightsOFF(leds)
# numTEST(led7,1)
# lightsOFF(leds)
# numTEST(led8,1)
# lightsOFF(leds)
# numTEST(led9,1)
# lightsOFF(leds)

# lightsOFF(leds3)
# lightTESTa(leds3,1,0)
# lightTESTa(leds3[::-1], 0.5,0)
# bit_GPIO(leds1,'0000001000')
# setupGPIO(leds1)
# bit_GPIO(leds1,"{0:b}".format(1023))
# time.sleep(20)
# bit_GPIO(leds1,"{0:b}".format(0))
# time.sleep(10)
# bit_GPIO(leds1,"{0:b}".format(1023))
# time.sleep(10)

# setupGPIO(leds1)
# lightsOFF(leds1)
# for counter in range(1,1024):
#     x = randint(0,1023)
#     # x = 1024 - counter
#     ss = "{0:b}".format(x)
#     print("{0:b}".format(x), x, len("{0:b}".format(x)))

#     bit_GPIO(leds3,ss)
#     time.sleep(30)
