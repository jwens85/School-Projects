#include <iostream>

// Declare CUDA functions with C linkage
extern "C" void run_cuda_kernel();
extern "C" void run_matrix_mul();

int main() {
    std::cout << "Launching hello_from_gpu..." << std::endl;
    run_cuda_kernel();

    std::cout << "Launching matrix multiplication..." << std::endl;
    run_matrix_mul();

    return 0;
}
