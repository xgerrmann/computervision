//#include <cuda/cuda.h>
// Cuda function definitions for funcitons in cudafuncs.cu
//__global__ void calcmap_cuda(xp_c, yp_c, wp_c, mxp_c, myp_c, mwp_c, h_c);

// ## Armadillo
#include <armadillo>
// ## eigen
#include <eigen3/Eigen/Dense> // << changes
arma::Cube<float> calcmapping(Eigen::Matrix3f Hi, int xmin_out, int ymin_out, int wmax, int hmax);
