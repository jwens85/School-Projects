cmake_minimum_required(VERSION 3.18)
project(parallel_histogram LANGUAGES CXX CUDA)

# Set C++ and CUDA standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Source files
add_executable(parallel_histogram
        main.cpp
        parallel_histogram.cu
)
