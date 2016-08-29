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

// ## Function Definitions
extern "C" void calcmapping(Eigen::MatrixXf *Mx, Eigen::MatrixXf *My,  Eigen::Matrix3f *Hi, int xmin_out, int ymin_out, int wmax, int hmax);

#ifndef _DEBUG
#define _DEBUG
#endif

#define _CUDAFUNCS_TIMEIT 0

#endif
