cmake_minimum_required(VERSION 2.8.4)
# About this project
#-------------------
project(src)

message("${PROJECT_NAME}")

# Add headers and source files here
#-------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)
#set(CUDA_SEPARABLE_COMPILATION OFF)

### CUDA
find_package(CUDA REQUIRED)
#list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;-O2;-DVERBOSE")
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_20;-std=c++11;")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

#add_subdirectory(cuda-utils)

# Create binaries
#-------------------
cuda_add_executable(homography	homography.cpp	homography.hpp			cudafuncs.cu	cudafuncs.hpp)
cuda_add_executable(main		main.cpp		main.hpp homography.hpp	cudafuncs.cu	cudafuncs.hpp)

# OPENCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# dlib
include(../include/dlib/cmake)

# Eigen3
# installed on the system

# Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

## tracker
set(ATTENTION_TRACKER_LIBRARY /home/xgerrmann/Documents/src/computervision/include/attention-tracker/build/libhead_pose_estimation.so)
find_library(${ATTENTION_TRACKER_LIBRARY} head_pose_estimation)
message("Head Pose library: ${ATTENTION_TRACKER_LIBRARY}")

## GPUtimer
set(GPUTIMER_LIBRARY /home/xgerrmann/Documents/src/computervision/lib/gputimer/lib/libgputimer.so)
find_library(${GPUTIMER_LIBRARY} gputimer)
message("GPUtimer library: ${GPUTIMER_LIBRARY}")

## OpenBLAS
set(OpenBLAS_LIBRARY /usr/lib/libopenblas.so)
find_library(${OpenBLAS_LIBRARY} openblas)
message("OpenBLASlibrary: ${OpenBLAS_LIBRARY}")

## Linker
##-------------------
# main
target_link_libraries(main			dlib	${OpenCV_LIBS}	${ARMADILLO_LIBRARIES}	${ATTENTION_TRACKER_LIBRARY}	openblas	${CUDA_LIBRARIES} gputimer)
# homography
target_link_libraries(homography	dlib	${OpenCV_LIBS}	${ARMADILLO_LIBRARIES}	${ATTENTION_TRACKER_LIBRARY}	openblas	${CUDA_LIBRARIES} gputimer)

option(DEBUG "DEBUG" OFF)
if(${DEBUG})
	message("DEBUG is ON")
	# definitions
	add_definitions(-D_DEBUG_)
endif()

## definitions
#add_definitions(-DTIMEIT)
