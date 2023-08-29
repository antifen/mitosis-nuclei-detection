import os
import shutil


if __name__ == '__main__':
    shutil.rmtree("/root/Code/mmdetection-master/HPF/test_patches")  # 强制删除该文件夹
    os.mkdir("/root/Code/mmdetection-master/HPF/test_patches")  # 重新创建新文件夹

    shutil.rmtree("/root/Code/mmdetection-master/HPF/test_results_HPF")
    os.mkdir("/root/Code/mmdetection-master/HPF/test_results_HPF")

    shutil.rmtree("/root/Code/mmdetection-master/HPF/test_results_index")
    os.mkdir("/root/Code/mmdetection-master/HPF/test_results_index")

    shutil.rmtree("/root/Code/mmdetection-master/HPF/test_patches_results")
    os.mkdir("/root/Code/mmdetection-master/HPF/test_patches_results")

    shutil.rmtree("/root/Code/mmdetection-master/HPF/test_results_patches_txt")
    os.mkdir("/root/Code/mmdetection-master/HPF/test_results_patches_txt")

    shutil.rmtree("/root/Code/mmdetection-master/HPF/class_image")
    os.mkdir("/root/Code/mmdetection-master/HPF/class_image")

    print("success clear")