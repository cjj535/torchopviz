"""
complex_graph.json is a list of tensors, torch operations and modules. Each element consist of: 
    id                  identity number for tensors and torch operations
    start_time          start timestamp(us)
    end_time            end timestamp(us)
    is_tensor           True for tensor
    is_leaf             True for tensor and torch operation
    label               module name/torch operation name/tensor shape
    parent              parent of torch op
    children            a children list of module, used to build trees of modules and torch ops
    next_nodes          a list of next nodes id, used to build graphs of tensors and torch ops
    info                lifetime, and metadata for tensors
You can generate your json file which can be displayed by "torchopviz".
"""
from torchopviz import offline_viz
offline_viz(file="./complex_graph.json")