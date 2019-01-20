import RPi.GPIO as GPIO
import time
import myGPIO
from random import randint

#Arrays of GPIO pins for the three orientations of prototype
# leds1 = [ 6,20,26, 3,21,5,15,14,2,16]
# leds2 = [15,14,3,2,21,20,16,5,26,6]
# leds3 = [16,5,2,26,21,14,6,20,3,15]
leds1 = [ 6,26,20, 5,21,3,16,2,14,15]
leds2 = [15,3,14,20,21,2,6,26,5,16]
leds3 = [16,2,5,14,21,26,15,3,20,6]
leds4 = [15,14,3,2,21,20,16,5,26,6]
leds5 = myGPIO.pinsGPIO#[5, 11, 9, 10, 22, 27, 17, 4, 3, 2]
led7 = myGPIO.segment7s


def setupGPIO(pins):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    for pin in pins:
        GPIO.setup(pin,GPIO.OUT)
        print ("setup Completed")

def lightsOFF(pins):
    for pin in pins:
        GPIO.output(pin,GPIO.LOW)
    print("ALL off")

# Turn on each light for x seconds
def lightTESTa(pins, wait1, wait2):
    for pin in pins:
        print("Pin ", pin, "is on")
        GPIO.output(pin,GPIO.HIGH)
        time.sleep(wait1)
        print("Pin ", pin, "is off")
        GPIO.output(pin, GPIO.LOW)
        time.sleep(wait2)

def bit_GPIO(pins, bits):
    while len(bits)<10:
        bits = "0"+bits
    for idx in range(0,len(bits)):
        print(idx,bits, bits[idx])
        if(bits[idx]=="1"):
             GPIO.output(pins[idx], GPIO.HIGH)
        else:
            GPIO.output(pins[idx], GPIO.LOW)
      

# *-

setupGPIO(leds5)
# lightsOFF(leds1)
# lightTESTa(leds1,1,0)
# lightTESTa(leds1[::-1], 0.5,0)
lightsOFF(leds5)
lightTESTa(leds5,1,0)
lightTESTa(leds5[::-1], 0.5,0)
lightsOFF(leds5)

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
