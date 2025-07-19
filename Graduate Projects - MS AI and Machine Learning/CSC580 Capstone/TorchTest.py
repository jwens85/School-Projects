import torch
import time

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device found.")

#GPU Stress Test
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nRunning stress test on {device}...")

size = 16384
start = time.time()

a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)
c = torch.randn(size, size, device=device)

x = a @ b
y = torch.sin(x) + torch.exp(c)
z = y @ b.T

torch.cuda.synchronize()
end = time.time()

print(f"Stress test completed in {end - start:.2f} seconds")

allocated = torch.cuda.memory_allocated(device) / 1024**3
reserved = torch.cuda.memory_reserved(device) / 1024**3
print(f"Memory allocated: {allocated:.2f} GB")
print(f"Memory reserved: {reserved:.2f} GB")
