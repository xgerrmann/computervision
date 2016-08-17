#
# This is a CMake makefile.  You can find the cmake utility and
# information about it at http://www.cmake.org
#

cmake_minimum_required(VERSION 2.8.4)

PROJECT(computervision)


# Tell CMake to compile a program.  We do this with the ADD_EXECUTABLE()
# statement which takes the name of the output executable and then a list of
# .cpp files to compile.  Here each example consists of only one .cpp file but
# in general you will make programs that const of many .cpp files.
ADD_EXECUTABLE(main src/main.cpp)
ADD_EXECUTABLE(homography src/homography.cpp)
# Then we tell it to link with dlib.
#TARGET_LINK_LIBRARIES(main dlib)


## Since there are a lot of examples I'm going to use a macro to simply this
## CMakeLists.txt file.  However, usually you will create only one executable in
## your cmake projects and use the syntax shown above.
#MACRO(add_file name)
#   ADD_EXECUTABLE(${name} ${name}.cpp)
#   TARGET_LINK_LIBRARIES(${name} dlib )
#ENDMACRO()

##here we apply our macros
#add_file(main)

include(include/dlib/cmake)
include_directories(include/eigen)
#add_subdirectory(include/attention_tracker)
#include_directories(${attention_tracker_SOURCE_DIR}))
#add_subdirectory(include/attention_tracker)
#add_library(attention_tracker /home/xander/computervision/include/attention-tracker/src/head_pose_estimation.hpp)
#add_library(attention_tracker include/attention-tracker/src/head_pose_estimation.hpp)
include_directories("${PROJECT_SOURCE_DIR}/include")
add_subdirectory(include/attention-tracker)

# Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# BLAS
#find_package(BLAS REQUIRED)
#include_directories(${BLAS_INCLUDE_DIRS})
# LAPACK
#find_package(LAPACK REQUIRED)
#include_directories(${LAPACK_INCLUDE_DIRS})
# OPENCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


# Linking
TARGET_LINK_LIBRARIES(main dlib attention_tracker ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(homography ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES})

##set(LAPACK_LIBDIRS "/usr/lib/liblapack")
##set(LAPACK_LIBRARIES lapack)
#
#if (OpenCV_FOUND)
#	include_directories(${OpenCV_INCLUDE_DIRS})
##   ADD_EXECUTABLE(webcam_face_pose_ex webcam_face_pose_ex.cpp)
#	TARGET_LINK_LIBRARIES(main dlib ${OpenCV_LIBS})
#	TARGET_LINK_LIBRARIES(homography ${OpenCV_LIBS})
#	TARGET_LINK_LIBRARIES(homography ${LAPACK_LIBS})
#else()
#	message("OpenCV NOT found :(")
#endif()
