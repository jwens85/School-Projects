#include <iostream>
#include <cuda_runtime.h>

// Forward declaration of the CUDA function
extern "C" void launchVectorAdd(const int* a, const int* b, int* c, int N);

const int N = 10;

int main() {
    std::cout << "We are now using CUDA over the GPU to add these vectors!\n";

    // Step 1: Allocate and initialize host (CPU) memory
    int h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 10;
    }

    // Step 2: Allocate device (GPU) memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_b, N * sizeof(int));
    cudaMalloc((void**)&d_c, N * sizeof(int));

    // Step 3: Copy inputs to device
    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Step 4: Launch kernel via host-callable wrapper
    launchVectorAdd(d_a, d_b, d_c, N);

    // Step 5: Copy result back to host
    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Step 6: Print result
    for (int i = 0; i < N; ++i) {
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << '\n';
    }

    // Step 7: Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
