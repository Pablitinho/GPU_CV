#Check the Cmake version
cmake_minimum_required(VERSION 2.8.9)


project (Example_GPU_CV)

add_subdirectory(src)
#------------------------------------------------------------------------------------------
#Project name
#------------------------------------------------------------------------------------------


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

message(STATUS $ENV{OPENCV_DIR})

#set(OpenCV_DIR ${OPENCV_DIR})
find_package(OpenCV REQUIRED )

find_package(CUDA REQUIRED)
SET(CUDA_PROPAGATE_HOST_FLAGS ON)

include_directories(${CUDA_TOOLKIT_INCLUDE})
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/x64/lib)

#includes 
SET(INCLUDE_GPU_CV_PATH ${CMAKE_CURRENT_LIST_DIR}/src/Gpu_module/include)
include_directories(${INCLUDE_GPU_CV_PATH})
message(STATUS ${INCLUDE_GPU_CV_PATH})

file(GLOB SOURCES_MAIN "src/*.cpp")
file(GLOB INCLUDE_MAIN "src/*.h")

#add exe
add_executable(${PROJECT_NAME} ${INCLUDE_MAIN} ${SOURCES_MAIN})

#set ( PROJECT_LINK_LIBS GPU_CV_LIB.dll )
#link_directories( ${CMAKE_SOURCE_DIR}/bin/)
	
#Add library 
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} GPU_CV_LIB)
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${PROJECT_LINK_LIBS} GPU_CV_LIB)
#link_directories(${CMAKE_SOURCE_DIR}/lib)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

