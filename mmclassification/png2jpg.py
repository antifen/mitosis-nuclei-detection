"""
此为将切分patch后的patch.png转化为jpg格式
"""

import os
import cv2 as cv


path = "/root/Code/mmclassification-master/data/test_data"
print(path)
os.chdir("/root/Code/mmclassification-master/data/test_data")
path_list = os.listdir(path)
print(path_list)

for filename in path_list:
    portion = os.path.splitext(filename)
    print('convert  ' + filename +'  to '+portion[0]+'.jpg')
    name = path + '/' + filename
    print(name)
    src = cv.imread(name)
    cv.imwrite(path +'/' + portion[0]+'.jpg', src)
for pic in path_list:
    if pic.endswith('.png'):
        os.remove(pic)

















