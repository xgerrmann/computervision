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
set(CUDA_SEPARABLE_COMPILATION OFF)

### CUDA
find_package(CUDA REQUIRED)
#if(CUDA_FOUND)
#	    list(APPEND CUDA_NVCC_FLAGS "-std=c++11;")
#		set(CUDA_SEPARABLE_COMPILCATION ON)
#		set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#endif()
## compile *.cu files
#CUDA_COMPILE(CU_O cudafuncs.cu)

#add_subdirectory(cuda-utils)

# Create binaries
#-------------------
#add_executable(homography	homography.cpp)
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
#set(ARMA /usr/local/lib/libarmadillo.so)
#include_directories(${ARMA})
if(${ARMADILLO_FOUND})
	#	message("Armadillo library found: ${ARMADILLO_INCLUDE_DIRS}")
	#	message("Armadillo library found: ${ARMADILLO_LIBRARIES}")
endif()

## tracker
set(ATTENTION_TRACKER_LIBRARY /home/xgerrmann/Documents/src/computervision/include/attention-tracker/build/libhead_pose_estimation.so)
find_library(${ATTENTION_TRACKER_LIBRARY} head_pose_estimation)
message("Head Pose library: ${ATTENTION_TRACKER_LIBRARY}")

## OpenBLAS
set(OpenBLAS_LIBRARY /usr/lib/libopenblas.so)
find_library(${OpenBLAS_LIBRARY} openblas)
message("OpenBLASlibrary: ${OpenBLAS_LIBRARY}")

## Linker
##-------------------
target_link_libraries(main			dlib	${OpenCV_LIBS}	${ARMADILLO_LIBRARIES}	${ATTENTION_TRACKER_LIBRARY}	openblas ${CUDA_LIBRARIES})


#target_link_libraries(homography	dlib	${OpenCV_LIBS}	${ARMADILLO_LIBRARIES}	${ATTENTION_TRACKER_LIBRARY}	openblas)
target_link_libraries(homography	dlib	${OpenCV_LIBS}	${ARMADILLO_LIBRARIES}	${ATTENTION_TRACKER_LIBRARY}	openblas	${CUDA_LIBRARIES})
#target_link_libraries(homography	dlib	${OpenCV_LIBS}	${ARMADILLO_LIBRARIES}	${ATTENTION_TRACKER_LIBRARY}	openblas	cuda-utils ${CUDA_LIBRARIES})

