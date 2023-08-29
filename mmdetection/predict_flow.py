"""
python tools/train.py configs/retinanet/retinanet_r50_fpn_1x_coco.py --gpu-ids=0
python tools/test.py configs/retinanet/retinanet_r50_fpn_1x_coco.py  work_dirs/retinanet_r50_fpn_1x_coco/epoch_20.pth --out results.pkl --eval mAP
"""

from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import predict_rely
import shutil
from PIL import Image
import json
import numpy as np
import openslide
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
thresh_patch_to_HPF = 0.3
NMSthreshold = 0.2
index_list = []

# 指定模型的配置文件和 checkpoint 文件路径
# config_file = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
# checkpoint_file = 'work_dirs/retinanet_r50_fpn_1x_coco-hospital/epoch_20.pth'

# config_file = 'configs/fsaf/fsaf_r50_fpn_1x_coco.py'
# checkpoint_file = 'work_dirs/fsaf_r50_fpn_1x_coco-hospital/epoch_20.pth'

# config_file = 'configs/fsaf/fsaf_r50_fpn_1x_coco.py'
# checkpoint_file = 'work_dirs/fsaf_r50_fpn_1x_coco-resnext-gn-2012data-2-CBAM-ThresholdReturnOld/epoch_20.pth'

# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_coco-hospital/epoch_20.pth'

# config_file = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
# checkpoint_file = 'work_dirs/retinanet_r50_fpn_1x_coco-hospital/epoch_20.pth'

# config_file = 'configs/ssd/ssd300_coco.py'
# checkpoint_file = 'work_dirs/ssd300_coco-hospital/epoch_20.pth'

# config_file = 'configs/yolo/yolov3_d53_320_273e_coco.py'
# checkpoint_file = 'work_dirs/yolov3_d53_320_273e_coco-hospital/epoch_23.pth'

config_file = 'configs/fsaf/fsaf_r50_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/fsaf_r50_fpn_1x_coco-resnext-gn-2012data-2-CBAM-ThresholdReturnOld/epoch_20.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# HPF地址
HPFs_list = os.listdir("/root/Code/mmdetection-master/HPF/test_HPF")
HPFs_path = "/root/Code/mmdetection-master/HPF/test_HPF"
test_patches_path = "/root/Code/mmdetection-master/HPF/test_patches"
test_patches_results_path = "/root/Code/mmdetection-master/HPF/test_patches_results"
test_results_HPF_path = "/root/Code/mmdetection-master/HPF/test_results_HPF"

for i in range(len(HPFs_list)):
    green_box_patches = {}
    # 首先切分这张HPF
    os.chdir("/root/Code/mmdetection-master/HPF")
    HPF_name = HPFs_list[i].split('.')[0]
    HPF_path = HPFs_path + "/" + HPFs_list[i]
    predict_rely.cut_HPF(HPFs_list[i], HPFs_path, test_patches_path)
    print("cut:" + HPFs_list[i])
    # 对每个patch进行测试，同时记录每个patch的pre_box坐标并存入字典,patch名为key，其三个list为value
    test_patches_list = os.listdir("/root/Code/mmdetection-master/HPF/test_patches")
    null_num = 0
    dict_patches_results = {}
    for j in range(len(test_patches_list)):
        patch_name = test_patches_list[j].split('.')[0]
        patch_path = test_patches_path + "/" + test_patches_list[j]
        img = patch_path
        # result是一个两层的array数组,[[x1, y1, x2, y2, 概率值]]
        result = inference_detector(model, img)
        print("test:" + test_patches_list[j])
        # 将result值分为三个数组存入字典green中
        predict_boxes = []
        predict_classes = ["mitosis"] * len(result[0])
        predict_scores = []
        for k in range(len(result[0])):
            tmp_xy = []
            tmp_xy.append(result[0][k][0])
            tmp_xy.append(result[0][k][1])
            tmp_xy.append(result[0][k][2])
            tmp_xy.append(result[0][k][3])
            predict_boxes.append(tmp_xy)
            predict_scores.append(result[0][k][4])
        if len(predict_boxes) == 0:
            # print("没有检测到任何目标!")
            null_num += 1
        else:
            # print("检测到有目标")
            # plt.imshow(original_img)
            # plt.show()
            green_box_patches[test_patches_list[j]] = predict_boxes, predict_classes, predict_scores
            # 保存预测的图片结果
            model.show_result(img, result, out_file=test_patches_results_path + "/" + test_patches_list[j])
    print("null_num:" + str(null_num))
    # 将patch的坐标变为HPF对应的新坐标。patch坐标参数存于green_box_patches中
    predict_boxes2, predict_classes2, predict_scores2 = predict_rely.patch_position_to_HPF(green_box_patches,
                                                                                           thresh_patch_to_HPF)
    # 此处还需进行一步NMS处理，现在可能出现的情况是一个GT可能对应多个pre_box，需要将score最高的那个pre_box留下，舍去其余的
    predict_boxes2, predict_classes2, predict_scores2 = predict_rely.my_NMS(predict_boxes2, predict_classes2,
                                                                            predict_scores2, NMSthreshold)

    HPF_result = []
    HPF_tmp_array = np.zeros(shape=(len(predict_boxes2), 5), dtype='float32')
    for p in range(HPF_tmp_array.shape[0]):
        HPF_tmp_array[p] = np.array([predict_boxes2[p][0], predict_boxes2[p][1], predict_boxes2[p][2],
                                     predict_boxes2[p][3], predict_scores2[p]])
    HPF_result.append(HPF_tmp_array)
    model.show_result(HPF_path, HPF_result, out_file=test_results_HPF_path + "/" + HPFs_list[i])

    # # 一张HPF全部完成后,要清空/test_patches与/test_results_patches两个文件夹
    shutil.rmtree("/root/Code/mmdetection-master/HPF/test_patches")  # 强制删除该文件夹
    os.mkdir("/root/Code/mmdetection-master/HPF/test_patches")  # 重新创建新文件夹
    shutil.rmtree("/root/Code/mmdetection-master/HPF/test_patches_results")
    os.mkdir("/root/Code/mmdetection-master/HPF/test_patches_results")

    # 将GT与Pre比较，20像素内算GT，输出一张HPF的TP等指标。
    xml_name = HPF_name.split(".")[0] + ".xml"
    mask_path = "/root/Code/mmdetection-master/HPF/test_xml" + "/" + xml_name
    gt_center_position = predict_rely.find_gt_center_position(mask_path)
    pr_center_position = predict_rely.find_pr_center_position(predict_boxes2)
    tmp_output_list = predict_rely.ouput_TP_FP(
        gt_center_position,
        pr_center_position,
        predict_boxes2)


    # 在计算指标之前，还需用分类网络进行分类，分类为true的才被认为TP
    # 分类网络需要将所有的predicted box输出成jpg以便分类
    slide = openslide.open_slide(HPF_path)
    for p in range(0, len(predict_boxes2)):
        cut_length = 256
        x_middle = int((predict_boxes2[p][0] + predict_boxes2[p][2]) / 2)
        y_middle = int((predict_boxes2[p][1] + predict_boxes2[p][3]) / 2)
        x_left_top = int(x_middle - (cut_length / 2))
        y_left_top = int(y_middle - (cut_length / 2))
        tile = np.array(
            slide.read_region((x_left_top, y_left_top), 0, (cut_length, cut_length)))
        tmp_image = Image.fromarray(tile)
        tmp_image_name = HPF_name.split('.')[0] + "_" + str(p) + ".png"
        if tmp_output_list[p] == 1:
            os.chdir("/root/Code/mmdetection-master/HPF/class_image2-test-hospital/T")
            tmp_image.save(tmp_image_name)
        else:
            os.chdir("/root/Code/mmdetection-master/HPF/class_image2-test-hospital/F")
            tmp_image.save(tmp_image_name)


