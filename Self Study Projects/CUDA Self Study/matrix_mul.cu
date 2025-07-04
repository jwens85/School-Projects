#include <cuda_runtime.h>
#include <cstdio>

#define N 2

// Matrix multiplication kernel (naive)
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int width) {
    int row = threadIdx.y;
    int col = threadIdx.x;

    float sum = 0;
    for (int k = 0; k < width; ++k) {
        sum += A[row * width + k] * B[k * width + col];
    }
    C[row * width + col] = sum;
}

// C-style linkage host function
extern "C" void run_matrix_mul() {
    float A[N * N] = {1, 2, 3, 4};
    float B[N * N] = {5, 6, 7, 8};
    float C[N * N] = {0};

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(float) * N * N);
    cudaMalloc(&d_B, sizeof(float) * N * N);
    cudaMalloc(&d_C, sizeof(float) * N * N);

    cudaMemcpy(d_A, A, sizeof(float) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(float) * N * N, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N, N);
    matrixMulKernel<<<1, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, sizeof(float) * N * N, cudaMemcpyDeviceToHost);

    printf("Matrix C =\n");
    for (int i = 0; i < N * N; ++i) {
        printf("%f ", C[i]);
        if ((i + 1) % N == 0) printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
