cmake_minimum_required(VERSION 3.18)
project(hello_cuda LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Enable CUDA as a language
enable_language(CUDA)

# Add source files
add_executable(hello_cuda main.cpp hello.cu)

# Link CUDA runtime statically
target_link_libraries(hello_cuda
        cudart_static
        kernel32
        user32
        gdi32
        winspool
        shell32
        ole32
        oleaut32
        uuid
        comdlg32
        advapi32
)
