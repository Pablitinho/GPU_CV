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
# Cuda 
find_package(CUDA REQUIRED)
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50)

set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION ON)
	
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

