#Check the Cmake version
cmake_minimum_required(VERSION 2.8.9)
#------------------------------------------------------------------------------------------
#Project name
#------------------------------------------------------------------------------------------
#project (GPU_CV_LIB)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_definitions(-Dshared_EXPORTS)
#add_definitions(-D__CUDACC__)
#add_definitions(-D__CUDACC_RTC__)
add_definitions(-D__GPU_ARCH__=610)

# Cuda 
find_package(CUDA REQUIRED)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50)

#set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-use_fast_math -G -g -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61")
#set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "-use_fast_math -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61")

set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-use_fast_math -G -g -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61")
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "-use_fast_math -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61")

#set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-use_fast_math -G -g -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60")
#set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} "-use_fast_math -gencode arch=compute_53,code=sm_53 -gencode arch=compute_60,code=sm_60")

set(CUDA_PROPAGATE_HOST_FLAGS ON)
enable_language(CUDA) 
#set(CUDA_SEPARABLE_COMPILATION ON)
	
#includes 
SET(INCLUDE_GPU_CV_PATH include)

include_directories(${INCLUDE_GPU_CV_PATH})
include_directories(${CUDA_TOOLKIT_INCLUDE})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/x64/lib)

file(GLOB SOURCES_GPU_CV "src/*.cpp" "src/*.cu")
file(GLOB INCLUDE_GPU_CV "include/*.h" "include/*.cuh")

cuda_add_library(GPU_CV_LIB SHARED ${SOURCES_GPU_CV} ${INCLUDE_GPU_CV})

#Add library 
target_link_libraries(GPU_CV_LIB ${CUDA_LIBRARIES})

install(TARGETS GPU_CV_LIB DESTINATION lib)

