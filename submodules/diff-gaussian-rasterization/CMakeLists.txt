#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

cmake_minimum_required(VERSION 3.20)

project(DiffRast LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

add_library(CudaRasterizer
    cuda_rasterizer/backward.h
    cuda_rasterizer/backward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/auxiliary.h
    # cuda_rasterizer/forward_with_contrib.h       
    # cuda_rasterizer/forward_with_contrib.cu
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/rasterizer.h
)

set_target_properties(CudaRasterizer PROPERTIES CUDA_ARCHITECTURES "70;75;86;120")

# Include directories
target_include_directories(CudaRasterizer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/cuda_rasterizer)
target_include_directories(CudaRasterizer PRIVATE third_party/glm ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# PyTorch
find_package(Torch REQUIRED)
target_include_directories(CudaRasterizer PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(CudaRasterizer "${TORCH_LIBRARIES}")

# Compile options
target_compile_options(CudaRasterizer PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -Xcompiler -fPIC>
)

