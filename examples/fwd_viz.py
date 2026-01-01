import torch
from torch import nn
from torchopviz import online_viz

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))
data = torch.randn(1,8)

online_viz(model, data, save_dir="./sample_data")