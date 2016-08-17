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
add_executable(homography homography.cpp)
#add_executable(${PROJECT_NAME} ${${PROJECT_NAME}_headers} ${${PROJECT_NAME}_sources})

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

# Linker
#-------------------
#target_link_libraries(${PROJECT_NAME} dlib)
#target_link_libraries(homography ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES})
#target_link_libraries(homography ${OpenCV_LIBS})
#target_link_libraries(homography ${OpenCV_LIBS})
target_link_libraries(homography ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES})

#######include_directories(include/eigen)
