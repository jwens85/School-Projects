#include <cuda_runtime.h>
#include <iostream>

#define cudaCheckError() {                                           \
cudaError_t e = cudaGetLastError();                              \
if (e != cudaSuccess) {                                          \
std::cerr << "CUDA Error: " << cudaGetErrorString(e) << std::endl; \
exit(EXIT_FAILURE);                                          \
}                                                                \
}

__global__ void histogram_kernel(const int* pixels, int num_pixels, int* histogram) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_pixels) {
        int val = pixels[idx];
        if (val >= 0 && val < 256) {
            atomicAdd(&histogram[val], 1);
        }
    }
}

// Add extern "C" for C++ linkage compatibility
extern "C"
void compute_histogram_gpu(const int* pixels, int num_pixels, int* histogram) {
    int* d_pixels = nullptr;
    int* d_histogram = nullptr;

    cudaMalloc(&d_pixels, num_pixels * sizeof(int));
    cudaMalloc(&d_histogram, 256 * sizeof(int));

    cudaMemcpy(d_pixels, pixels, num_pixels * sizeof(int), cudaMemcpyHostToDevice);

    // Zero out the histogram on device
    cudaMemset(d_histogram, 0, 256 * sizeof(int));

    int blockSize = 256;
    int numBlocks = (num_pixels + blockSize - 1) / blockSize;
    histogram_kernel<<<numBlocks, blockSize>>>(d_pixels, num_pixels, d_histogram);

    cudaDeviceSynchronize();
    cudaCheckError();

    cudaMemcpy(histogram, d_histogram, 256 * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
    cudaFree(d_histogram);
}
