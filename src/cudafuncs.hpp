//cudafuncs.hpp
#ifndef cudafuncs_h
#define cudafuncs_h

#include <stdlib.h>
#include <iostream>

//// ## Timer
//#include "../lib/timer/timer.hpp"

// ## Armadillo
#include <armadillo>

// ## Eigen
 #include <eigen3/Eigen/Dense>

// ## GPUtimer
//#include "../lib/gputimer/gputimer.hpp"

// ## openCV cuda
//#include <opencv2/core/cuda.hpp>

// ## openCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/core/devmem2d.hpp>

//## cuda
#include <cuda_runtime.h>

#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

// ## Function Definitions
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number);
void calcmapping(Eigen::MatrixXi& Mx, Eigen::MatrixXi& My,  Eigen::Matrix3f& Hi, float xmin, float ymin);
void domapping(const cv::Mat& image_input, cv::Mat& image_output, Eigen::MatrixXi& Mx, Eigen::MatrixXi& My, float x_map_min, float y_map_min);
void copy(const cv::Mat& image_in, cv::Mat& image_out);

#ifdef _DEBUG_
#define _CUDAFUNCS_DEBUG 1
#endif
#define _CUDAFUNCS_DEBUG 1

#define _CUDAFUNCS_TIMEIT 0

#endif
