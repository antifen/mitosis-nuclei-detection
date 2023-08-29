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
thresh_patch_to_HPF = 0.3
NMSthreshold = 0.2
index_list = []

# config_file = 'configs/retinanet/retinanet_r50_fpn_1x_coco.py'
# checkpoint_file = 'work_dirs/retinanet_r50_fpn_1x_coco/epoch_19.pth'

# config_file = 'configs/fsaf/fsaf_r50_fpn_1x_coco.py'
# checkpoint_file = 'work_dirs/fsaf_r50_fpn_1x_coco/epoch_20.pth'

config_file = 'configs/fsaf/fsaf_r50_fpn_1x_coco.py'
checkpoint_file = 'work_dirs/fsaf_r50_fpn_1x_coco-resnext-gn-2012data-2-CBAM-ThresholdReturnOld/epoch_20.pth'

# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = 'work_dirs/faster_rcnn_r50_fpn_1x_coco-2012-table5/epoch_17.pth'

# config_file = 'configs/yolo/yolov3_d53_320_273e_coco.py'
# checkpoint_file = 'work_dirs/yolov3_d53_320_273e_coco/epoch_37.pth'

# config_file = 'configs/ssd/ssd300_coco.py'
# checkpoint_file = 'work_dirs/ssd300_coco/epoch_10.pth'

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

    # 在计算指标之前，还需用分类网络进行分类，分类为true的才被认为TP
    # 分类网络需要将所有的predicted box输出成jpg以便分类
    slide = openslide.open_slide(HPF_path)
    for p in range(0, len(predict_boxes2)):
        cut_length = 112
        x_middle = int((predict_boxes2[p][0] + predict_boxes2[p][2]) / 2)
        y_middle = int((predict_boxes2[p][1] + predict_boxes2[p][3]) / 2)
        x_left_top = int(x_middle - (cut_length / 2))
        y_left_top = int(y_middle - (cut_length / 2))
        tile = np.array(
            slide.read_region((x_left_top, y_left_top), 0, (cut_length, cut_length)))
        tmp_image = Image.fromarray(tile)
        tmp_image_name = HPF_name.split('.')[0] + "_" + str(p) + ".png"
        os.chdir("/root/Code/mmdetection-master/HPF/class_image")
        tmp_image.save(tmp_image_name)

    # 分类后，对predict_boxes2的每一个元素进行判断，记住标注不为1的元素，并将其删除，从而形成最终的预测结果。
    # 读取文件名为index.txt的文件，里边包含了所有分类后不是核分裂像的image名，根据名字找到对应的HPF和在predicted_boxes的序号
    # 具体做法为，找到此一轮循环HPF名称对应的image，并记住其序号，最后在predicted_boxes2中删除
    # tmp_result = []
    # tmp_result_num = []
    # os.chdir("/root/Code/retinaNet")
    # with open('index.txt', 'r') as f:
    #     for line in f:
    #         tmp_result.append(line.strip('\n'))
    # for q in range(0, len(tmp_result)):
    #     tmp_name = tmp_result[q].split('_')[0] + "_" + tmp_result[q].split('_')[1]
    #     tmp_num = tmp_result[q].split('_')[-1]
    #     # print(tmp_name, tmp_num)
    #     if tmp_name == HPF_name.split(".")[0]:
    #         tmp_result_num.append(int(tmp_num))
    # # 降序排列，防止删除list元素时出现问题
    # tmp_result_num = sorted(tmp_result_num, reverse=True)
    # # 删除分类后错误的box
    # predict_boxes2, predict_classes2, predict_scores2 = predict_rely.classification(predict_boxes2,
    #                                                                                 predict_classes2,
    #                                                                                 predict_scores2, tmp_result_num)

    # 将GT与Pre比较，20像素内算GT，输出一张HPF的TP等指标。
    xml_name = HPF_name.split(".")[0] + ".xml"
    mask_path = "/root/Code/mmdetection-master/HPF/test_xml" + "/" + xml_name
    gt_center_position = predict_rely.find_gt_center_position(mask_path)
    pr_center_position = predict_rely.find_pr_center_position(predict_boxes2)
    TP, pr_num, gt_num, Precision, Recall, Fscore, less_num, more_num = predict_rely.count_index(
        gt_center_position,
        pr_center_position)

    # 将结果写入index.txt文件
    tmp_index_list = []
    tmp_name_distance = 43 - len(HPF_name)
    str1 = "name:" + HPF_name + tmp_name_distance * ' ' + '   '
    # tmp_TP_distance = 10 - len(str(TP))
    str2 = "TP:" + str(round(TP, 2)) + '   '
    # tmp_pr_distance = 10 - len(str(pr_num))
    str3 = "pr_num:" + str(round(pr_num, 2)) + '   '
    # tmp_gt_distance = 10 - len(str(gt_num))
    str4 = "gt_num:" + str(round(gt_num, 2)) + '   '
    # tmp_Pre_distance = 10 - len(str(Precision))
    str5 = "Precision:" + str(round(Precision, 2)) + '   '
    # tmp_Re_distance = 10 - len(str(Recall))
    str6 = "Recall:" + str(round(Recall, 2)) + '   '
    # tmp_F_distance = 10 - len(str(Fscore))
    str7 = "Fscore:" + str(round(Fscore, 2)) + '   '
    # tmp_less_distance = 10 - len(str(less_num))
    str8 = "less_num:" + str(round(less_num, 2)) + '   '
    # tmp_more_distance = 10 - len(str(more_num))
    str9 = "more_num:" + str(round(more_num, 2)) + '   '
    tmp_index_list.append(str1)
    tmp_index_list.append(str2)
    tmp_index_list.append(str3)
    tmp_index_list.append(str4)
    tmp_index_list.append(str5)
    tmp_index_list.append(str6)
    tmp_index_list.append(str7)
    tmp_index_list.append(str8)
    tmp_index_list.append(str9)
    index_list.append(tmp_index_list)
    # 最后清空字典，方便下一章HPF使用
    green_box_patches.clear()
    print("--end--")
    # break

print(index_list)
f = open("/root/Code/mmdetection-master/HPF/test_results_index/index.txt", "w")
for i in range(0, len(index_list)):
    for j in range(0, len(index_list[i])):
        f.write(index_list[i][j])
    f.write("\n")
f.close()
TP_num = 0
Pr_num = 0
GT_num = 0
for i in range(0, len(index_list)):
    TP_num += int(index_list[i][1][3:8])
    Pr_num += int(index_list[i][2][7:12])
    GT_num += int(index_list[i][3][7:12])
print("TP_num:{}".format(TP_num))
print("Pr_num:{}".format(Pr_num))
print("GT_num:{}".format(GT_num))

