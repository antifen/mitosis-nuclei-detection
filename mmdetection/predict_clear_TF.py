import os
import shutil


if __name__ == '__main__':
    shutil.rmtree("/root/Code/mmdetection-master/HPF/class_image2-test-hospital/F")  # 强制删除该文件夹
    os.mkdir("/root/Code/mmdetection-master/HPF/class_image2-test-hospital/F")  # 重新创建新文件夹

    shutil.rmtree("/root/Code/mmdetection-master/HPF/class_image2-test-hospital/T")
    os.mkdir("/root/Code/mmdetection-master/HPF/class_image2-test-hospital/T")


    print("success clear")


# import os
# import glob
# import shutil
#
# filePath = '/root/Code/mmdetection-master/HPF/class_image2/F'
# newFilePath = '/root/Code/mmclassification-master/data/my_dataset/val/mitosis-1'
#
# filename = os.listdir(filePath)
# for i in filename:
#     shutil.copy(filePath + '/' + i, newFilePath + '/' + i)
#     print("success:", i)


