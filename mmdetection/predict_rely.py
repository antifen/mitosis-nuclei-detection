from xml.dom import minidom
import math
import numpy as np
import cv2
import os
import collections
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont



# def file_name(root_path, picturetype):
#     filename = []
#     for root, dirs, files in os.walk(root_path):
#         for file in files:
#             if os.path.splitext(file)[1] == picturetype:
#                 filename.append(os.path.join(root, file))
#     return filename


# 横着切分HPF
def cut_HPF(HPF_name, path_name, out_path):
    # filename = file_name(path_name, ".jpg")  # 得到文件夹内所有图片的 “路径+文件名”     RGB
    path = path_name + '/' + HPF_name
    name = HPF_name.split(".")[0]
    dst = np.zeros((2084 + 60, 2084 + 60, 3))  # 补上边缘之后的HPF      RGB       2144-2084=60
    img = cv2.imread(path, 1)  # RGB
    dst[:2084, :2084] = img  # dst中复制原HPF
    dst_small = np.zeros((224, 224, 3))  # 小图尺寸
    num = 0
    last_top_x = 0
    last_top_y = 0
    for i in range(11):
        for j in range(11):
            last_top_y = j * 192
            dst_small = dst[last_top_x:(last_top_x + 224), last_top_y:(last_top_y + 224)]
            cv2.imwrite(out_path + '/' + '{}_{}.jpg'.format(name, num), dst_small)
            num += 1
        last_top_x += 192
    # print(num)


# 将patches_results拼接成原来的HPF，暂时没有用处
def patch_combination(name, path1, path2):
    patches_list = os.listdir(path1)
    dst = np.zeros((2084 + 60, 2084 + 60, 3))  # 首先新建一个空图片
    for j in range(0, len(patches_list)):
        path = "/root/Code/ssd/test_results_patches/" + patches_list[j]
        tmp_img = cv2.imread(path, 1)  # RGB
        num = patches_list[j].split("_")[-1]
        num = num.split(".")[0]
        num = int(num)  # str->int
        pos_y = num % 11
        pos_x = math.floor(num / 11)
        dst[(192 * pos_x):(192 * pos_x + 224), (192 * pos_y):(192 * pos_y + 224)] = tmp_img
    dst = dst[:2084, :2084]
    name = name
    tmp_path = path2 + '/'
    cv2.imwrite(tmp_path + '{}'.format(name), dst)
    # print("success!!")


# patch的坐标变为HPF对应的新坐标，参照combination函数
def patch_position_to_HPF(dict, thresh123):
    predict_boxes = []
    predict_classes = []
    predict_scores = []
    for key in dict:
        num = str(key).split("_")[-1]
        num = num.split(".")[0]
        num = int(num)  # str->int
        pos_x = num % 11
        pos_y = math.floor(num / 11)
        x_offset = pos_x*192
        y_offset = pos_y*192
        for i in range(0, len(dict[key][0])):
            dict[key][0][i][0] = dict[key][0][i][0] + x_offset
            dict[key][0][i][1] = dict[key][0][i][1] + y_offset
            dict[key][0][i][2] = dict[key][0][i][2] + x_offset
            dict[key][0][i][3] = dict[key][0][i][3] + y_offset

    for key in dict:
        for p in range(0, len(dict[key][0])):
            predict_boxes.append(dict[key][0][p])
        for q in range(0, len(dict[key][1])):
            predict_classes.append(dict[key][1][q])
        for r in range(0, len(dict[key][2])):
            predict_scores.append(dict[key][2][r])
    predict_boxes = np.array(predict_boxes)
    predict_classes = np.array(predict_classes)
    predict_scores = np.array(predict_scores)
    predict_boxes, predict_classes, predict_scores = change_threshold(predict_boxes, predict_classes, predict_scores, thresh123)
    return predict_boxes, predict_classes, predict_scores


# 筛去predict_scores中小于0.5的pre_box（预测0.7为分界点结果较好）
def change_threshold(predict_boxes, predict_classes, predict_scores, thresh123):
    predict_boxes = predict_boxes.tolist()
    predict_classes = predict_classes.tolist()
    predict_scores = predict_scores.tolist()
    predict_boxes2 = []
    predict_classes2 = []
    predict_scores2 = []
    for i in range(0, len(predict_scores)):
        if predict_scores[i] > thresh123:
            predict_boxes2.append(predict_boxes[i])
            predict_classes2.append(predict_classes[i])
            predict_scores2.append(predict_scores[i])
    predict_boxes2 = np.array(predict_boxes2)
    predict_classes2 = np.array(predict_classes2)
    predict_scores2 = np.array(predict_scores2)
    return predict_boxes2, predict_classes2, predict_scores2


# 根据HPF的xml文件求出中心点坐标，参考my_faster_rcnn的predict代码
def find_gt_center_position(xml_path):
    xy_list = []
    dom = minidom.parse(xml_path)
    root = dom.documentElement

    x1 = root.getElementsByTagName('xmin')
    y1 = root.getElementsByTagName('ymin')
    x2 = root.getElementsByTagName('xmax')
    y2 = root.getElementsByTagName('ymax')
    for i in range(0, len(x1)):
        xx1 = x1[i]
        xx1 = int(xx1.firstChild.data)
        yy1 = y1[i]
        yy1 = int(yy1.firstChild.data)
        xx2 = x2[i]
        xx2 = int(xx2.firstChild.data)
        yy2 = y2[i]
        yy2 = int(yy2.firstChild.data)
        xx = (xx1 + xx2) / 2
        yy = (yy1 + yy2) / 2
        xy_list.append([xx, yy])

    return xy_list


# 求出pre_box的中心点坐标
def find_pr_center_position(predicted_boxes):
    center_list = []
    if len(predicted_boxes):
        for i in predicted_boxes:
            xx = (i[0] + i[2]) / 2
            yy = (i[1] + i[3]) / 2
            center_list.append([xx, yy])
    return center_list

def ouput_TP_FP(gt_center_position, pr_center_position, predict_boxes2):
    gt_num = len(gt_center_position)
    pr_num = len(pr_center_position)
    result_ij = []
    tmp_list = [0] * len(predict_boxes2)

    for i in range(0, len(gt_center_position)):
        min_distance = 1000
        distance = 0
        tmp_ij = 0
        # 对于每一个i-gt，找到其对应的最小距离j-pr之后
        for j in range(0, len(pr_center_position)):
            xy = np.array(gt_center_position[i]) - np.array(pr_center_position[j])
            distance = math.hypot(xy[0], xy[1])
            if distance <= min_distance:
                min_distance = distance
                tmp_ij = (i, j)
        if min_distance < 20:
            result_ij.append(tmp_ij)
            tmp_j = tmp_ij[1]
            tmp_list[tmp_j] = 1

    return tmp_list


# 算出TP等指标
def count_index(gt_center_position, pr_center_position):
    gt_num = len(gt_center_position)
    pr_num = len(pr_center_position)
    result_ij = []
    for i in range(0, len(gt_center_position)):
        min_distance = 1000
        distance = 0
        tmp_ij = 0
        # 对于每一个i-gt，找到其对应的最小距离j-pr之后，记录并在j的集合中去掉该j
        for j in range(0, len(pr_center_position)):
            xy = np.array(gt_center_position[i]) - np.array(pr_center_position[j])
            distance = math.hypot(xy[0], xy[1])
            if distance <= min_distance:
                min_distance = distance
                tmp_ij = (i, j)
        if min_distance < 20:
            result_ij.append(tmp_ij)

    TP = len(result_ij)
    if TP == 0:
        Precision = 0
    else:
        Precision = TP / pr_num

    if TP == 0:
        Recall = 0
    else:
        Recall = TP / gt_num

    if TP == 0:
        Fscore = 0
    else:
        Fscore = (2 * Precision * Recall) / (Precision + Recall)
    less_num = gt_num - TP
    more_num = pr_num - TP
    # print("gt_center_position:" + str(gt_center_position))
    # print("pr_center_position:" + str(pr_center_position))
    # print(result_ij)
    # print(TP, pr_num, gt_num, Precision, Recall, Fscore, less_num, more_num)

    return TP, pr_num, gt_num, Precision, Recall, Fscore, less_num, more_num


# 尝试将有pre_box的patch记录，并在拼接的HPF上重新画一遍patch，但结果发现没有区别。。。
# def patch_combination_again(list, name):
#     # 首先根据name找到HPF并转化为数组
#     patches_list = list
#     path = "/root/Code/faster_rcnn/test_results_HPF" + '/' + name
#     HPF = cv2.imread(path, 1)
#     dst = np.zeros((2084 + 60, 2084 + 60, 3))  # 首先新建一个空图片
#     dst[:2084, :2084] = HPF
#     # 接着吧list中的patch重新画一遍
#     for j in range(0, len(patches_list)):
#         path = "/root/Code/faster_rcnn/test_results_patches/" + patches_list[j]
#         tmp_img = cv2.imread(path, 1)  # RGB
#         num = patches_list[j].split("_")[-1]
#         num = num.split(".")[0]
#         num = int(num)  # str->int
#         pos_y = num % 11
#         pos_x = math.floor(num / 11)
#         dst[(192 * pos_x):(192 * pos_x + 224), (192 * pos_y):(192 * pos_y + 224)] = tmp_img
#     dst = dst[:2084, :2084]
#     tmp_path = "/root/Code/faster_rcnn/test_results_HPF/"
#     cv2.imwrite(tmp_path + '{}'.format(name), dst)
#     print("---success---")

# my_NMS需要的函数,计算IoU：
def count_iou(list1, list2):
    x11 = max(list1[0], list2[0])
    y11 = max(list1[1], list2[1])
    x22 = min(list1[2], list2[2])
    y22 = min(list1[3], list2[3])
    area1 = (list1[2] - list1[0]) * (list1[3] - list1[1])
    area2 = (list2[2] - list2[0]) * (list2[3] - list2[1])
    w = max(0, x22 - x11)
    h = max(0, y22 - y11)
    overlaps = w * h
    IoU = overlaps/(area1 + area2 - overlaps)
    return IoU


# 手动实现NMS，原因在于目前会存在1个GT对应多个Pre_box的现象，需要手动保留最大score的pre_box，筛去其他的
def my_NMS(predicted_boxes, predicted_classes, predicted_scores, nmsthreshold):
    predicted_boxes = list(predicted_boxes)
    predicted_classes = list(predicted_classes)
    predicted_scores = list(predicted_scores)
    tmp_list = []
    for i in range(0, len(predicted_boxes)):
        for j in range(0, len(predicted_boxes)):
            tmp_iou = 0
            if i != j:
                tmp_iou = count_iou(predicted_boxes[i], predicted_boxes[j])
            if tmp_iou >= nmsthreshold:
                if predicted_scores[i] > predicted_scores[j]:
                    tmp_list.append(j)
                else:
                    tmp_list.append(i)
    tmp_list = list(set(tmp_list))
    tmp_list.sort(reverse=True)
    for i in range(0, len(tmp_list)):
        predicted_boxes.pop(tmp_list[i])
        predicted_classes.pop(tmp_list[i])
        predicted_scores.pop(tmp_list[i])
    predicted_boxes = np.array(predicted_boxes)
    predicted_classes = np.array(predicted_classes)
    predicted_scores = np.array(predicted_scores)

    return predicted_boxes, predicted_classes, predicted_scores


def classification(predicted_boxes, predicted_classes, predicted_scores, tmp_list):
    predicted_boxes = list(predicted_boxes)
    predicted_classes = list(predicted_classes)
    predicted_scores = list(predicted_scores)
    for i in range(0, len(tmp_list)):
        del predicted_boxes[tmp_list[i]]
        del predicted_classes[tmp_list[i]]
        del predicted_scores[tmp_list[i]]
    predicted_boxes = np.array(predicted_boxes)
    predicted_classes = np.array(predicted_classes)
    predicted_scores = np.array(predicted_scores)

    return predicted_boxes, predicted_classes, predicted_scores








