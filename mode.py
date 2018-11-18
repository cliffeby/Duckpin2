import statistics
import collections
from array import *
nbits = [0]*10
allBits = [0]*10, [0]*10, [0]*10, [0]*10, [0] * \
    10, [0]*10, [0]*10, [0]*10, [0]*10, [0]*10
pinCounts = collections.deque(3*[1023], 10)
print(pinCounts, type(pinCounts))
pinCounts.append(1021)
pinCounts.append(1021)
pinCounts.append(1021)
print(pinCounts, type(pinCounts))
pinCounts.append(512)
pinCounts.append(512)
pinCounts.append(512)
print(pinCounts, type(pinCounts))
pinCounts.append(512)
pinCounts.append(512)
pinCounts.append(512)
print(pinCounts, type(pinCounts))
print(pinCounts[0], pinCounts[1])
print(allBits)
for idx, pinCount in enumerate(pinCounts):

    bits = "{0:b}".format(pinCount)
    print(bits)
    while len(bits) < 10:
        bits = "0"+bits
    for idx2 in range(0, len(bits)):
        allBits[idx2][idx] = bits[idx2]
print(allBits)
pinCount = 0
for i in range(0, len(pinCounts)):
    nbits[i] = statistics.mode(allBits[i])
    pinCount = pinCount + 2**(9-i)
    print(nbits[i])
print('Mode Pincount', pinCount)
