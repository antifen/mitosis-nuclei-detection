from mmcls.datasets import MyDataset
from mmcls.apis import init_model, inference_model, show_result_pyplot
import mmcv
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# config_file = 'configs/swin_transformer/swin-base_16xb64_in1k.py'
# checkpoint_file = 'work_dirs/swin-base_16xb64_in1k/epoch_72.pth'

# config_file = 'configs/resnet/resnet34_8xb32_in1k.py'
# checkpoint_file = 'work_dirs/resnet34_8xb32_in1k-hospital-50-98/epoch_76.pth'

# config_file = 'configs/resnet/resnet152_8xb32_in1k.py'
# checkpoint_file = 'work_dirs/resnet152_8xb32_in1k-2012-table5/epoch_85.pth'

# config_file = 'configs/vgg/vgg16_8xb32_in1k.py'
# checkpoint_file = 'work_dirs/vgg16_8xb32_in1k/epoch_5.pth'

config_file = 'configs/resnet/resnet34_8xb32_in1k.py'
checkpoint_file = 'work_dirs/resnet34_8xb32_in1k-2012-best-second/epoch_89-best.pth'

# 根据配置文件和 checkpoint 文件构建模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Add an attribute for visualization convenience
model.CLASSES = MyDataset.CLASSES

images_list = "/root/Code/mmclassification-master/data/test_data"
# images_list = "/root/Code/mmdetection-master/HPF/class_image2-test-hospital/F"
img_list = os.listdir(images_list)
f = open("/root/Code/mmclassification-master/data/index.txt", "w")
# f = open("/root/Code/mmdetection-master/HPF/index.txt", "w")

for i in range(len(img_list)):
    tmp_path = images_list + "/" + img_list[i]
    tmp_result = inference_model(model, tmp_path)
    print(img_list[i] + "  ", tmp_result['pred_label'])
    if tmp_result['pred_label'] == 1:
        f.write(img_list[i].split('.')[0])
        f.write("\n")
f.close()


# # single image test
# img = 'data/test_data/test.jpg'
# result = inference_model(model, img)
# print(result)
# print(result['pred_class'])
# show_result_pyplot(model, img, result)