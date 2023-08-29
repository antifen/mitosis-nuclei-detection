# optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='Adam', lr=0.0001, eps=1e-08, amsgrad=False)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[100])
runner = dict(type='EpochBasedRunner', max_epochs=100)
