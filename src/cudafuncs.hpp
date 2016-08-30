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

// ## openCV
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/devmem2d.hpp>

// ## Function Definitions
void calcmapping(Eigen::MatrixXf *Mx, Eigen::MatrixXf *My,  Eigen::Matrix3f *Hi, int xmin_out, int ymin_out, int wmax, int hmax);
void domapping(cv::Mat& image_output, const cv::Mat& image_input, Eigen::MatrixXf *Mx, Eigen::MatrixXf *My);

#ifdef _DEBUG_
#define _CUDAFUNCS_DEBUG 1
#endif
#define _CUDAFUNCS_DEBUG 1

#define _CUDAFUNCS_TIMEIT 0

#endif
