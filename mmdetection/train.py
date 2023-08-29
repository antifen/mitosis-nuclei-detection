# from mmdet.apis import init_detector, inference_detector
#
# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# # 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# # 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# device = 'cuda:0'
# # 初始化检测器
# model = init_detector(config_file, checkpoint_file, device=device)
# # 推理演示图像
# inference_detector(model, 'demo/demo.jpg')


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

#config_file = 'configs/ssd/ssd300_coco.py'
#checkpoint_file = 'work_dirs/ssd300_coco/epoch_10.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# HPF地址
HPFs_list = os.listdir("/root/Code/mmdetection-master/HPF/test_HPF")
HPFs_path = "/root/Code/mmdetection-master/HPF/test_HPF"
test_patches_path = "/root/Code/mmdetection-master/HPF/test_patches"
test_patches_results_path = "/root/Code/mmdetection-master/HPF/test_patches_results"
test_results_HPF_path = "/root/Code/mmdetection-master/HPF/test_results_HPF"

print(1)
print(2)
print(model.summary)

