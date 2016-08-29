//cudafuncs.hpp
#ifndef cudafuncs_h
#define cudafuncs_h

//// ## Timer
//#include "../lib/timer/timer.hpp"

// ## Armadillo
#include <armadillo>

// ## Eigen
#include <eigen3/Eigen/Dense>

// ## GPUtimer
#include "../lib/gputimer/gputimer.hpp"

// ## openCV cuda
#include <opencv2/core/cuda.hpp>

// ## Function Definitions
extern "C" void calcmapping(Eigen::MatrixXf *Mx, Eigen::MatrixXf *My,  Eigen::Matrix3f *Hi, int xmin_out, int ymin_out, int wmax, int hmax);

extern "C" void domapping(cv::cuda::GpuMat *image_out, cv::cuda::GpuMat *image_in, Eigen::MatrixXf *Mx, Eigen::MatrixXf *My);

#ifdef _DEBUG_
#define _CUDAFUNCS_DEBUG 1
#endif

#define _CUDAFUNCS_TIMEIT 0

#endif
