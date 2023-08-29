_base_ = [
    '../_base_/models/vgg16.py',
    '../_base_/datasets/my_dataset.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]
# _base_ = [
#     '../_base_/models/vgg16.py',
#     '../_base_/datasets/imagenet_bs32_pil_resize.py',
#     '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
# ]
optimizer = dict(lr=0.01)