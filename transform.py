import torch
import torch.nn as nn
from projekt import Fruits360CnnModel

model = torch.load("nowymodel.pt")
print(model)
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(128,3,100,100)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# Save the TorchScript model
traced_script_module.save("nowymodelandroid.pt")