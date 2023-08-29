# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(type='VGG', depth=16, num_classes=2),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=512,
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0, class_weight=torch.tensor([1/215, 1/3123]).cuda()),
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 2),
    ))


# model original settings
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(type='VGG', depth=16, num_classes=2),
#     neck=None,
#     head=dict(
#         type='ClsHead',
#         loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#         topk=(1, 2),
#     ))
