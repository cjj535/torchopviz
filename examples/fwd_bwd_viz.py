import time
import os

import torch
from torch import nn
import torch.profiler as profiler

from torchopviz import launch_visualizer, init, load_data


def generate_data(model, input, label):
    if not torch.cuda.is_available():
        return
    device_str = "cuda:0"
    device = torch.device(device_str)

    model = model.to(device)

    def trace_handler(prof: torch.profiler.profile):
        prof._memory_profile()

    # Profiler
    prof = profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA
        ],
        on_trace_ready=trace_handler,
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    )

    prof.start()
    output = model(input.to(device))
    label = label.to(device)
    loss = torch.nn.functional.cross_entropy(output, label)
    loss.backward()
    torch.cuda.synchronize()
    prof.stop()


# Initialize model
model = nn.Sequential()
model.add_module('W0', nn.Linear(1024, 2048, bias=False))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(2048, 10, bias=False))
model = model.to()

# Initialize data
data = torch.randn(1024, 1024)
label = torch.randn(1024, 10)

# Init torchopviz
init(save_dir=".", save_backward=True)

# Generate data
generate_data(model, data, label)

# Launch server
json_data = load_data("./complex_graph.json")
launch_visualizer(json_data, port=5000, threshold=1.0)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("stopped")