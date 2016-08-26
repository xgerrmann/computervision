cmake_minimum_required(VERSION 2.8)

set(sources cudafuncs.cu)

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

cuda_add_library(cuda-utils STATIC ${sources})

target_link_libraries(cuda-utils ${ARMADILLO_LIBRARIES} ${CUDA_LIBRARIES})
