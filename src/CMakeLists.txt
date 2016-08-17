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

# Eigen
include_directories(../include/eigen)

# Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# tracker
include_directories(../include/attention-tracker)

# Linker
#-------------------
target_link_libraries(main dlib ${OpenCV_LIBS} head_pose_estimation)
target_link_libraries(homography ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES})

#######include_directories(include/eigen)
