//cudafuncs.hpp
#ifndef cudafuncs_h
#define cudafuncs_h

// ## Armadillo
#include <armadillo>

// ## Eigen
#include <eigen3/Eigen/Dense>

// ## Function Definitions
extern "C" arma::Cube<float> calcmapping(Eigen::Matrix3f Hi, int xmin_out, int ymin_out, int wmax, int hmax);

#endif
