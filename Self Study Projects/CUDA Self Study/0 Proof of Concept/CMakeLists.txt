cmake_minimum_required(VERSION 3.18)
project(hello_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

enable_language(CUDA)

add_executable(hello_cuda main.cpp hello.cu matrix_mul.cu)

# Link against CUDA runtime
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
