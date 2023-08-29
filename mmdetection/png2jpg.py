"""
此为将切分patch后的patch.png转化为jpg格式
"""

import os
import cv2 as cv


# path = "/root/Code/mmclassification-master/data/my_dataset/train/class1"
# print(path)
# os.chdir("/root/Code/mmclassification-master/data/my_dataset/train/class1")
# path_list = os.listdir(path)
# print(path_list)

# path = "/root/Code/mmclassification-master/data/my_dataset/val/mitosis"
# # path = "/root/Code/RetinaNet-CBAM/FP"
# print(path)
# os.chdir("/root/Code/mmclassification-master/data/my_dataset/val/mitosis")
# # os.chdir("/root/Code/RetinaNet-CBAM/FP")

# path = "/root/Code/mmdetection-master/HPF/class_image2-test-hospital-yolo/T"
# print(path)
# os.chdir("/root/Code/mmdetection-master/HPF/class_image2-test-hospital-yolo/T")
# path_list = os.listdir(path)
# print(path_list)

# path = "/root/Code/mmdetection-master/HPF/class_image2-test-hospital-yolo/F"
# print(path)
# os.chdir("/root/Code/mmdetection-master/HPF/class_image2-test-hospital-yolo/F")
# path_list = os.listdir(path)
# print(path_list)



path = "/root/Code/mmdetection-master/HPF/class_image"
print(path)
os.chdir("/root/Code/mmdetection-master/HPF/class_image")
path_list = os.listdir(path)
print(path_list)

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

















# file_list = os.listdir('./summation/wsi-1')
# # print(file_list)
# os.chdir('./summation/wsi-1')
# num = len(file_list)
#
# for i in range(num):
#     old_name = os.path.join("E:\Mitosis\Dataset\trasnform", file_list[i])
#     a, b = os.path.split(file_list[i])
#     if b != '.jpg':
#         new_name = os.path.join("E:\Mitosis\Dataset\trasnform", (a + '.jpg'))
# #2. 删除其他格式
# for pic in file_list:
#     if pic.endswith('.png'):
#         os.remove(pic)

