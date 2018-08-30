path = '/dp/log/firstFile.txt'
mytxt = '1,2,3'
file = open(path,'w')
file.write(mytxt)
file.close
file = open(path,'r')
dd = file.read()
print("DD",dd)
import threading
import time
from datetime import datetime
def hello():
    global timesup
    timesup = False
    print ("hello, world", timesup)
    
timesup = True
t = threading.Timer(3.0, hello)
t.start() # after 30 seconds, "hello, world" will be printed
print ('timer is running', timesup)
td = datetime.now()
print (td)
td.ctime()
print ('dp '+ td.ctime()+ "_" +str(1053) + "_" +str(1))