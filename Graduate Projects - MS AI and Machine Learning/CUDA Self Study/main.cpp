#include <iostream>

// Declare the function as C linkage
extern "C" void run_cuda_kernel();

int main() {
    std::cout << "Launching CUDA kernel...\n";
    run_cuda_kernel();
    std::cout << "Finished.\n";
    return 0;
}
