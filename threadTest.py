import threading
import time
import random


def trip():
    print('looking for ball')
    i = 0
    try:
        while True:
            if random.random() > .90:
                i = i+1
                print("Balllllllll", i)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass


t = threading.Thread(target=trip, name='thread1')
t.start()

print('Statred')
try:
    while True:
        print('Still running')
        time.sleep(0.1)
except KeyboardInterrupt:
    t._stop()
    pass
