#include <cuda_runtime.h>
#include <cstdio>

// Kernel function that runs on the GPU
__global__ void hello_kernel() {
    printf("Hello from GPU!\n");
}

// Expose to C++ without name mangling
extern "C" void run_cuda_kernel() {
    hello_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
}
