cmake_minimum_required(VERSION 3.27)
project(DCN_V3)
set(CMAKE_PREFIX_PATH D:/libtorch/libtorch/share/cmake/Torch)

set(CMAKE_CXX_STANDARD 14)
include_directories(ops_dcnv3/src)
include_directories(ops_dcnv3/src/cpu)
include_directories(ops_dcnv3/src/cuda)
find_package( Torch REQUIRED )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(DCN_V3
        ops_dcnv3/src/cpu/dcn1Dv3.h
        ops_dcnv3/src/cpu/dcn1Dv3_cpu.cpp
        ops_dcnv3/src/cpu/dcn1Dv3_cpu.h
        ops_dcnv3/src/cpu/dcn2Dv3_cpu.cpp
        ops_dcnv3/src/cpu/dcn2Dv3_cpu.h
        ops_dcnv3/src/cpu/dcn3Dv3_cpu.cpp
        ops_dcnv3/src/cpu/dcn3Dv3_cpu.h
        ops_dcnv3/src/cuda/dcn1Dv3_cuda.cu
        ops_dcnv3/src/cuda/dcn1Dv3_cuda.h
        ops_dcnv3/src/cuda/dcn1Dv3_im2col_cuda.cuh
        ops_dcnv3/src/cuda/dcn2Dv3_cuda.cu
        ops_dcnv3/src/cuda/dcn2Dv3_cuda.h
        ops_dcnv3/src/cuda/dcn2Dv3_im2col_cuda.cuh
        ops_dcnv3/src/cuda/dcn3Dv3_cuda.cu
        ops_dcnv3/src/cuda/dcn3Dv3_cuda.h
        ops_dcnv3/src/cuda/dcn3Dv3_im2col_cuda.cuh
        ops_dcnv3/src/dcn2Dv3.h
        ops_dcnv3/src/dcn3Dv3.h
        ops_dcnv3/src/vision.cpp
        ops_dcnv3/src/dcn1Dv3.h)
