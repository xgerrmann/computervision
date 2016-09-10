cmake_minimum_required(VERSION 2.8)
# About this project
#-------------------
project(src)

message("${PROJECT_NAME}")

# Add headers and source files here
#-------------------
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(BUILD_SHARED_LIBS OFF)

### CUDA
find_package(CUDA REQUIRED)

# Create binaries
#-------------------
#cuda_add_executable(homography	homography.cpp	cudafuncs.cu)
#cuda_add_executable(test test.cpp	cudafuncs.cu)

# main script
cuda_add_executable(main		main.cpp		cudafuncs.cu)
# test for comp speed vs image size
cuda_add_executable(speedtest	speedtest.cpp	cudafuncs.cu)
# test for computational speed distribution
cuda_add_executable(distrotest	comp_distribution.cpp	cudafuncs.cu)


# OPENCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# dlib
include(../include/dlib/cmake)

# Eigen3
# installed on the system

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
target_link_libraries(main			dlib	${OpenCV_LIBS}	${ATTENTION_TRACKER_LIBRARY}	openblas	${CUDA_LIBRARIES} gputimer)
# speedtest
target_link_libraries(speedtest		dlib	${OpenCV_LIBS}	${ATTENTION_TRACKER_LIBRARY}	openblas	${CUDA_LIBRARIES} gputimer)
# comp distribution test
target_link_libraries(distrotest	dlib	${OpenCV_LIBS}	${ATTENTION_TRACKER_LIBRARY}	openblas	${CUDA_LIBRARIES} gputimer)
# homography
#target_link_libraries(homography	dlib ${OpenCV_LIBS}	${CUDA_LIBRARIES} gputimer)

option(DEBUG "DEBUG" OFF)
if(${DEBUG})
	message("DEBUG is ON")
	# definitions
	add_definitions(-D_DEBUG_)
endif()

## definitions
#add_definitions(-DTIMEIT)
