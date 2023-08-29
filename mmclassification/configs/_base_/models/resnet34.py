import torch
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=34,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./work_dirs/resnet34.pth',
            prefix='backbone',
        )),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=torch.tensor([1/215, 1/3123]).cuda()),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 2),
    ))
