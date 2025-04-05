import torch
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define matrix size (large)
MATRIX_SIZE = 40_000

# Create two large random matrices
A = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
B = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)

# Benchmark CPU
A_cpu = A.to("cpu")
B_cpu = B.to("cpu")
start_time = time.time()
result_cpu = torch.matmul(A_cpu, B_cpu)
end_time = time.time()
print(f"CPU time: {end_time - start_time:.4f} seconds")

# Benchmark GPU
torch.cuda.synchronize()  # Ensure GPU is ready
start_time = time.time()
result_gpu = torch.matmul(A, B)  # Matrix multiplication on GPU
torch.cuda.synchronize()  # Wait for completion
end_time = time.time()
print(f"GPU time: {end_time - start_time:.4f} seconds")

# Verify the computation (optional)
print(f"Output shape: {result_gpu.shape}")
