import torch
import torchvision
from collections import OrderedDict

model = torchvision.models.resnet34(pretrained=True)
state_dict = OrderedDict()

for k, v in model.state_dict().items():
    if not k.startswith('fc'):
        state_dict['backbone.' + k] = v
    else:
        state_dict['head.' + k] = v

torch.save(state_dict, './work_dirs/resnet34.pth')