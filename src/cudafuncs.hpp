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
void calcmapping(Eigen::MatrixXf& Mx, Eigen::MatrixXf& My,  Eigen::Matrix3f& Hi, int width_in, int height_in, float xmin, float ymin);
void domapping(const cv::Mat& image_input, cv::Mat& image_output, Eigen::MatrixXf& Mx, Eigen::MatrixXf& My);
void copy(const cv::Mat& image_in, cv::Mat& image_out);

#ifdef _DEBUG_
#define _CUDAFUNCS_DEBUG 1
#endif
#define _CUDAFUNCS_DEBUG 1

#define _CUDAFUNCS_TIMEIT 0

#endif
