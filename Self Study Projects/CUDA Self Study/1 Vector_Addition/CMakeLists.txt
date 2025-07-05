cmake_minimum_required(VERSION 3.18)

project(vector_add_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Make sure this path is correct
set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")

# Include headers and link directories explicitly
include_directories("${CUDAToolkit_ROOT}/include")
link_directories("${CUDAToolkit_ROOT}/lib/x64")

# Enable CUDA language
enable_language(CUDA)

add_executable(vector_add_gpu
        main.cpp
        vector_add.cu
)

# Link to CUDA runtime libraries
target_link_libraries(vector_add_gpu
        cudart_static
)
