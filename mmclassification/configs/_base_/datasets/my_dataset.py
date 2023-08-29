# # 医院数据集配置
# dataset_type = 'MyDataset'
# classes = ['mitosis', 'mitosis-1']  # 数据集中各类别的名称
#
# # img_norm_cfg = dict(
# #     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#
# img_norm_cfg = dict(
#     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
#
# albu_train_transforms = [
#     dict(
#         type='VerticalFlip',
#         p=0.5),
#     dict(
#         type='HorizontalFlip',
#         p=0.5),
#     dict(
#         type='RandomRotate90',
#         p=0.5),
#     dict(
#         type='GaussianBlur',
#         p=0.5),
#     dict(
#         type='MedianBlur',
#         p=0.5),
#     dict(
#         type='RandomBrightnessContrast',
#         p=0.5),
#     dict(
#         type='RandomGamma',
#         p=0.5),
#     # dict(
#     #     type='ShiftScaleRotate',
#     #     shift_limit=0.0625,
#     #     scale_limit=0.0,
#     #     rotate_limit=30,
#     #     interpolation=2,
#     #     p=0.5),
# ]
#
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='RandomResizedCrop', size=128),
#     dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
#     # dict(type='Albu',
#     #      transforms = albu_train_transforms,
#     #      keymap={'img': 'image'}),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='ToTensor', keys=['gt_label']),
#     dict(type='Collect', keys=['img', 'gt_label'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', size=(64, -1)),
#     dict(type='CenterCrop', crop_size=32),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='ImageToTensor', keys=['img']),
#     dict(type='Collect', keys=['img'])
# ]
#
# data = dict(
#     samples_per_gpu=32,
#     workers_per_gpu=2,
#     train=dict(
#         type=dataset_type,
#         data_prefix='data/my_dataset/train',
#         ann_file='data/my_dataset/meta/train.txt',
#         classes=classes,
#         pipeline=train_pipeline
#     ),
#     val=dict(
#         type=dataset_type,
#         data_prefix='data/my_dataset/val',
#         ann_file='data/my_dataset/meta/val.txt',
#         classes=classes,
#         pipeline=test_pipeline
#     ),
#     test=dict(
#         type=dataset_type,
#         data_prefix='data/my_dataset/val',
#         ann_file='data/my_dataset/meta/val.txt',
#         classes=classes,
#         pipeline=test_pipeline
#     )
# )
# evaluation = dict(interval=1, metric='f1_score', metric_options={'topk': (1, ), 'average_mode': 'none'})









# 2012 参数配置
dataset_type = 'MyDataset'
classes = ['mitosis', 'mitosis-1']  # 数据集中各类别的名称

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

albu_train_transforms = [
    dict(
        type='VerticalFlip',
        p=0.5),
    dict(
        type='HorizontalFlip',
        p=0.5),
    dict(
        type='RandomRotate90',
        p=0.5),
    dict(
        type='GaussianBlur',
        p=0.5),
    dict(
        type='MedianBlur',
        p=0.5),
    dict(
        type='RandomBrightnessContrast',
        p=0.5),
    dict(
        type='RandomGamma',
        p=0.5),
    # dict(
    #     type='ShiftScaleRotate',
    #     shift_limit=0.0625,
    #     scale_limit=0.0,
    #     rotate_limit=30,
    #     interpolation=2,
    #     p=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=128),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    # dict(type='Albu',
    #      transforms = albu_train_transforms,
    #      keymap={'img': 'image'}),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/my_dataset/train',
        ann_file='data/my_dataset/meta/train.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/my_dataset/val',
        ann_file='data/my_dataset/meta/val.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/my_dataset/val',
        ann_file='data/my_dataset/meta/val.txt',
        classes=classes,
        pipeline=test_pipeline
    )
)
evaluation = dict(interval=1, metric='f1_score', metric_options={'topk': (1, ), 'average_mode': 'none'})


#2012 参数配置-vgg16要求出入size为224x224
dataset_type = 'MyDataset'
classes = ['mitosis', 'mitosis-1']  # 数据集中各类别的名称

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_norm_cfg = dict(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(224, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='data/my_dataset/train',
        ann_file='data/my_dataset/meta/train.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='data/my_dataset/val',
        ann_file='data/my_dataset/meta/val.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='data/my_dataset/val',
        ann_file='data/my_dataset/meta/val.txt',
        classes=classes,
        pipeline=test_pipeline
    )
)
evaluation = dict(interval=1, metric='f1_score', metric_options={'topk': (1, ), 'average_mode': 'none'})