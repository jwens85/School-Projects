import torch

# Allocate a tensor on the GPU
tensor = torch.randn(3, 3).to("cuda")
print(f"Tensor on GPU:\n{tensor}")
