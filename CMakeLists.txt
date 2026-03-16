cmake_minimum_required(VERSION 3.18)

project(CUDAInferenceEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_executable(inference_engine
    src/main.cpp
    src/matrix_mul.cu
)