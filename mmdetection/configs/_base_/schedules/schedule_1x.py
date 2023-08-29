# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=0.0001,
    gamma=0.33,
    step=[3, 6, 9, 12, 15, 18])
runner = dict(type='EpochBasedRunner', max_epochs=20)

# CosineAnnealing
# Cyclic
