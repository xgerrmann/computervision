#cmake_minimum_required(VERSION 2.8)
project(cuda-utils)

set(sources cudafuncs.cu cudafuncs.hpp)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})
if(${ARMADILLO_FOUND})
	#	message("Armadillo library found: ${ARMADILLO_INCLUDE_DIRS}")
	#	message("Armadillo library found: ${ARMADILLO_LIBRARIES}")
endif()

## CUDA
find_package(CUDA REQUIRED)
if(CUDA_FOUND)
	#	list(APPEND CUDA_NVCC_FLAGS "-std=c++11;")
	#	set(CUDA_SEPARABLE_COMPILCATION ON)
	#	set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endif()
# compile *.cu files
#CUDA_COMPILE(CU_O cudafuncs.cu)
#set(CUDA_NVCC_FLAGS "-arch=sm_20 -rdc=true -lcudadevrt")
set(CUDA_NVCC_FLAGS "-arch=sm_20 -lcudadevrt")

#cuda_add_library(cuda-utils STATIC ${sources})
cuda_add_library(${PROJECT_NAME} SHARED ${sources})

target_link_libraries(${PROJECT_NAME} ${ARMADILLO_LIBRARIES} ${CUDA_LIBRARIES})
