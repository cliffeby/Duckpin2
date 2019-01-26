# GPIO>BCM settings

# 7-segment
segment7s = [2, 3, 4, 14, 15, 17, 18]

segment7_0 = []  # [1,2,3,4,5,6]
segment7_1 = []  # [3,4]
segment7_2 = []  # [0,1,3,5,6]
segment7_3 = []  # [0,3,4,5,6]
segment7_4 = []  # [0,2,4,6]
segment7_5 = []  # [0,2,3,4,5]
segment7_6 = []  # [0,1,2,3,4]
segment7_7 = []  # [4,5,6]
segment7_8 = []  # [0,1,2,3,4,5,6]
segment7_9 = []  # [0,2,3,4,5,6]
# 0
for seg in [1, 2, 3, 4, 5, 6]:
    segment7_0.append(segment7s[seg])
# 1
for seg in [3,4]:
    segment7_1.append(segment7s[seg])
# 2
for seg in [0,1,4,6,]:
    segment7_2.append(segment7s[seg])
# 3
for seg in [0,3,4,5,6]:
    segment7_3.append(segment7s[seg])
# 4
for seg in [0,2,4,6]:
    segment7_4.append(segment7s[seg])
# 5
for seg in [0,2,3,4,5]:
    segment7_5.append(segment7s[seg])
# 6
for seg in [0,1,2,3,4]:
    segment7_6.append(segment7s[seg])
# 7
for seg in [4,5,6]:
    segment7_7.append(segment7s[seg])
# 8
for seg in [0,1,2,3,4,5,6]:
    segment7_8.append(segment7s[seg])
# 9
for seg in [0,2,3,4,5,6]:
    segment7_9.append(segment7s[seg])


segment7All = [segment7_0, segment7_1, segment7_2, segment7_3,
               segment7_4, segment7_5, segment7_6, segment7_7, segment7_8, segment7_9]
# print(segment7_0, segment7_4,segment7_8)
# Pin leds

pinsGPIO = [7,5,6,12,13,19,16,26,20,21]
#  Laser trip-wire GPIO
sensor = [23,22,27]



# *****************************************************************************************
#  GPIO>BOARD SETTINGS

# # 7-segemnt
# segment7s = [3,5,7,8,10,11,12]
# segment7_0 = [5,7,8,10,11,12]#[8, 24, 23, 15, 7, 25]
# segment7_1 = [10,12]#[8, 24]
# segment7_2 = [3,5,8,11,12]#[8, 23, 15, 25, 14]
# segment7_3 = [3,8,10,11,12]#[8, 24, 23, 25, 14]
# segment7_4 = [3,7,10,12]#[8, 24, 7, 14]
# segment7_5 = [3,7,8,10,11]#[24, 23, 7, 25, 14]
# segment7_6 = [3,5,7,8,10]#[24, 23, 15, 7, 14]
# segment7_7 = [10,11,12]#[8, 24, 25]
# segment7_8 = [3,5,7,8,10,11,12]#[8, 24, 23, 15, 7, 25, 14]
# segment7_9 = [3,7,8,10,11,12]#[8, 24, 23, 7, 25, 14]
# segment7All = [segment7_0, segment7_1, segment7_2, segment7_3,
#                segment7_4, segment7_5, segment7_6, segment7_7, segment7_8, segment7_9]

# Pin leds
# pinsGPIO = [26, 29, 31, 32, 33, 35, 36, 37, 38, 40]

#  Laser trip-wire GPIO
# sensor = [15,16,13]
