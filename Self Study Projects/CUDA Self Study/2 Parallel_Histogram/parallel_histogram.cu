#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void histogram_kernel(const int* d_pixels, int* d_histogram, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        atomicAdd(&d_histogram[d_pixels[idx]], 1);
    }
}

extern "C"
void compute_histogram_gpu(const std::vector<int>& pixels, std::vector<int>& histogram) {
    int* d_pixels;
    int* d_histogram;
    int num_pixels = static_cast<int>(pixels.size());

    // Allocate device memory
    cudaMalloc(&d_pixels, num_pixels * sizeof(int));
    cudaMemcpy(d_pixels, pixels.data(), num_pixels * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_histogram, 256 * sizeof(int));
    cudaMemset(d_histogram, 0, 256 * sizeof(int));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;
    histogram_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_pixels, d_histogram, num_pixels);

    // Copy result back
    cudaMemcpy(histogram.data(), d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_pixels);
    cudaFree(d_histogram);
}
