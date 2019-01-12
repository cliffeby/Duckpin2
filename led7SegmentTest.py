import RPi.GPIO as GPIO
import time
import threading
from random import randint
from gpiozero import LightSensor


# Arrays of GPIO pins for the three orientations of prototype

# leds = [18, 24, 25, 7, 8, 23, 9, 4]
# led0 = [18, 24, 25, 7, 8, 23]
# led1 = [18, 24]
# led2 = [18, 25, 7, 23, 9]
# led3 = [18, 24, 25, 23, 9]
# led4 = [18, 24, 8, 9]
# led5 = [24, 25, 8, 23, 9]
# led6 = [24, 25, 7, 8, 9]
# led7 = [18, 24, 23]
# led8 = [18, 24, 25, 7, 8, 23, 9]
# led9 = [18, 24, 8, 23, 9]
# ledAll = [led0, led1, led2, led3, led4, led5, led6, led7, led8, led9]


leds = [8,24,23,15,7,25,14]
led0 = [8,24,23,15,7,25]
led1 = [8,24]
led2 = [8,23,15,25,14]
led3 = [8,24,23,25,14]
led4 = [8,24,7,14]
led5 = [24,23,7,25,14]
led6 = [24,23,15,7,14]
led7 = [8,24,25]
led8 = [8,24,23,15,7,25,14]
led9 = [8,24,23,7,25,14]
ledAll = [led0, led1, led2, led3, led4, led5, led6, led7, led8, led9]


def setupGPIO(pins):
    GPIO.setmode(GPIO.BCM)
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


def trip():
    GPIO.setup(14, GPIO.OUT)
    GPIO.output(14, GPIO.LOW)
    time.sleep(.5)
    GPIO.setup(14, GPIO.IN)
    light = 0
    count = 0
    timesup = True
    t = threading.Timer(3.0, timeout)
    while True:

        if (GPIO.input(14) == GPIO.LOW):
            if (light == 0):
                # print('light = 0 and input low')
                continue
            print('no ball')
            light = 0
        else:
            if timesup == False:
                print('light = 1 and input high')
                continue

            print('BALLLLLLL ', count)
            lightsOFF(leds)
            GPIO.output((ledAll[count % 10]), GPIO.LOW)
            light = 1
            count = count + 1
            time.sleep(.5)
            timesup = True


def sensor():
    ldr = LightSensor(4)
    NOTLDR = LightSensor(14)
    while True:
        print(ldr.value, NOTLDR.value)

# def bit_GPIO(pins, bits):
#     while len(bits)<10:
#         bits = "0"+bits
#     for idx in range(0,len(bits)):
#         print(idx,bits, bits[idx])
#         if(bits[idx]=="1"):
#              GPIO.output(pins[idx], GPIO.HIGH)
#         else:
#             GPIO.output(pins[idx], GPIO.LOW)


# *-


setupGPIO(leds)
lightsOFF(ledAll)
lightTESTa(ledAll,.1,0)
# lightTESTa(leds[::1], 10.5,0)
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
