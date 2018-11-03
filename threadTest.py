import threading
import time
import random


def trip():
    print('looking for ball')
    while True:
        if random.random() > .95:
            print("Balllllllll")
        time.sleep(0.1)


t = threading.Thread(target=trip, name='thread1')
t.start()

print('Statred')
while True:
    print('Still running')
    time.sleep(5.0)
