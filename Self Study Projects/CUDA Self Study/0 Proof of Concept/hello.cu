#include <cuda_runtime.h>
#include <cstdio>

// Simple GPU kernel
__global__ void hello_from_gpu() {
    printf("Hello from GPU thread %d\n", threadIdx.x);
}

// C-style linkage for the host function
extern "C" void run_cuda_kernel() {
    hello_from_gpu<<<1, 5>>>();
    cudaDeviceSynchronize();
}
