#include <cuda_runtime.h>

// CUDA kernel for element-wise vector addition
__global__ void vectorAdd(const int* a, const int* b, int* c, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

// Expose the CUDA function to C++ with C linkage
extern "C" void launchVectorAdd(const int* a, const int* b, int* c, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();
}
