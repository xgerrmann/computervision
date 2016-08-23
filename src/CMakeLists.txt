cmake_minimum_required(VERSION 2.8.4)
# About this project
#-------------------
project(src)

message("${PROJECT_NAME}")

# Add headers and source files here
#-------------------
#set (${PROJECT_NAME}_headers ./main.hpp ./homography.hpp)
#set (${PROJECT_NAME}_sources ./main.cpp ./homography.cpp)
#set (${PROJECT_NAME}_headers ./homography.hpp)
#set (${PROJECT_NAME}_sources ./homography.cpp)

# Create binaries
#-------------------
add_executable(homography	homography.cpp)
add_executable(main			main.cpp)

# OPENCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# dlib
include(../include/dlib/cmake)

# Eigen3
# installed on the system
#find_package(Eigen3 REQUIRED)
#include_directories(EIGEN3_INCLUDE_DIR)
#include_directories(../include/eigen)

# Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
if(${ARMADILLO_FOUND})
	message("Armadillo library found: ${ARMADILLO_INCLUDE_DIRS}")
	message("Armadillo library found: ${ARMADILLO_LIBRARIES}")
endif()

## tracker
#include_directories(../include/attention-tracker)
#set(ATTENTION_TRACKER_INCLUDE_DIRS /home/xgerrmann/Documents/src/computervision/include/attention-tracker/build/)
set(ATTENTION_TRACKER_LIBRARY /home/xgerrmann/Documents/src/computervision/include/attention-tracker/build/libhead_pose_estimation.so)
#set(ATTENTION_TRACKER_LIBRARY_LOCATION /home/xgerrmann/Documents/src/computervision/include/attention-tracker/build)
find_library(${ATTENTION_TRACKER_LIBRARY} head_pose_estimation)
#message("Head Pose library location: ${ATTENTION_TRACKER_LIBRARY_LOCATION}")
message("Head Pose library: ${ATTENTION_TRACKER_LIBRARY}")
#include_directories(${ATTENTION_TRACKER_INCLUDE_DIRS})

# Linker
#-------------------
#target_link_libraries(main dlib ${OpenCV_LIBS} head_pose_estimation)
#target_link_libraries(homography ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES} head_pose_estimation)
target_link_libraries(main dlib ${OpenCV_LIBS} ${ATTENTION_TRACKER_LIBRARY})
target_link_libraries(homography ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES} ${ATTENTION_TRACKER_LIBRARY})

