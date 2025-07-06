// CUDA kernel for element-wise vector addition

#include <cuda_runtime.h>
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
// --------------------------------------------------------------------------------------------------
// CUDA Vector Addition Module - vector_add.cu
// --------------------------------------------------------------------------------------------------
//
// This file defines both the CUDA kernel and its launch wrapper function used to execute a simple
// vector addition operation in parallel on the GPU.
//
// --------------------------------------------------------------------------------------------------
// Overview of Functionality
// --------------------------------------------------------------------------------------------------
//
// 1. __global__ void vectorAddKernel(...):
//    - This is a device kernel function launched by the host.
//    - Each GPU thread computes the sum of one element from input vectors d_a and d_b.
//    - The result is stored in the corresponding index of output vector d_c.
//
// 2. void launchVectorAdd(...):
//    - This is a host function callable from main.cpp.
//    - It calculates an appropriate grid configuration for launching the kernel.
//    - It ensures the kernel runs on the GPU and synchronizes device execution before returning.
//
// --------------------------------------------------------------------------------------------------
// Kernel Design
// --------------------------------------------------------------------------------------------------
//
// - The grid-stride loop design uses the formula:
//      int i = blockIdx.x * blockDim.x + threadIdx.x;
//   which computes a global index for each thread based on its block and thread ID.
// - This approach allows simple parallelization of loop-style array operations.
//
// - The if-condition (i < n) ensures threads don't access out-of-bounds memory,
//   which can occur when the number of threads exceeds the array length.
//
// --------------------------------------------------------------------------------------------------
// Thread & Block Configuration
// --------------------------------------------------------------------------------------------------
//
// - Threads are grouped into blocks, and blocks are grouped into a grid.
// - Here, threadsPerBlock is set to 256, which is a commonly used value for optimal GPU occupancy.
// - The number of required blocks is computed using ceiling division to ensure full coverage.
//
//     Example:
//         For n = 1000, threadsPerBlock = 256,
//         blocksPerGrid = ceil(1000 / 256) = 4
//
// - CUDA will launch (blocksPerGrid × threadsPerBlock) total threads to cover all elements of the input arrays in parallel.
//
// --------------------------------------------------------------------------------------------------
// Synchronization
// --------------------------------------------------------------------------------------------------
//
// - cudaDeviceSynchronize() ensures the kernel has completed before control returns to the host.
// - Without synchronization, subsequent CPU operations might begin before GPU computation finishes.
//
// --------------------------------------------------------------------------------------------------
// Memory Assumptions
// --------------------------------------------------------------------------------------------------
//
// - Input and output arrays (d_a, d_b, d_c) must already be allocated on GPU memory using cudaMalloc.
// - These arrays must be populated (d_a, d_b) before calling launchVectorAdd(...).
//
// --------------------------------------------------------------------------------------------------
// Integration with Host Code
// --------------------------------------------------------------------------------------------------
//
// - This .cu file is compiled and linked with the host main.cpp using CMake.
// - The host allocates memory, copies data to device, calls launchVectorAdd(), and copies results back.
//
// --------------------------------------------------------------------------------------------------
// End of Documentation for vector_add.cu
// --------------------------------------------------------------------------------------------------
