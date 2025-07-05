//
// Created by jwens on 7/5/2025.
//

#include <iostream>
#include <vector>

int main() {
    const int N = 5;
    std::vector<int> a{1, 2, 3, 4, 5};
    std::vector<int> b{10, 20, 30, 40, 50};
    std::vector<int> c(N);

    std::cout << "Starting Vector Addition (host-side only):\n";

    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
        std::cout << a[i] << " + " << b[i] << " = " << c[i] << '\n';
    }

    std::cout << "This version ran on CPU only. CUDA integration coming next.\n";

    return 0;
}
// ----------------------
// Lessons Learned Summary
// ----------------------

// This was our first working C++ program using std::vector to perform CPU-side vector addition.
// It outputs each element-wise sum to the terminal using std::cout.
// - We didn't just copy/paste code here, we took the time to learn why the syntax is the way it is
// - We declared main() to return an int, which is standard in C++.
// - Returning 0 from main() signals successful execution to the operating system or build tools.
// - We used const int N = 5 to define the vector size as a constant. const makes N immutable.
// - std::vector<int> creates a dynamic array (vector) of integers from the C++ Standard Library.
// - a, b, and c are vectors. a and b were initialized with values, c was pre-allocated with default zeros.
// - Vector memory is automatically managed—metadata lives on the stack, values live on the heap.
// - We used a for-loop with (int i = 0; i < N; ++i), which is analogous to Python's range(N).
// - In each loop iteration, we performed a[i] + b[i] and stored the result in c[i].
// - We printed the result using std::cout, which uses the << operator for output.
// - \n is an escape character that creates a newline. '\\n' prints the literal \n.
// - std::endl can also be used to print a newline but also flushes the output buffer.
// - :: is the scope resolution operator; std::vector accesses the vector class inside the std namespace.
// - Indentation isn't syntactically required in C++, but it's critical for human readability.
// - We learned how string literals are stored in read-only memory, while vectors are managed dynamically.
// - We discussed stack vs heap memory, fragmentation, and why the vector abstraction matters.

// - Our CMakeLists.txt file defines project settings and is essential for build automation with CMake.
// - It tells the build system what compilers to use, what language standards to apply, and what source files to compile.
// - The executable and object files (.exe and .obj) are placed inside cmake-build-debug by default.

// - We are now ready to move from CPU-side computation to GPU-side computation using CUDA.
// - This will involve writing .cu files, allocating device memory with cudaMalloc, copying data with cudaMemcpy,
//   and launching a kernel to parallelize vector addition across CUDA threads.

// -- Next up: CUDA-izing this program.
