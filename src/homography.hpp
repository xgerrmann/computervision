#include <iostream>

//#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "paths.hpp"
#include <math.h>
#include <chrono>
#include <ctime>
// ## Eigen
#include <Eigen/Dense>
#include <limits.h> // for max values of datatypes
// ## Armadillo
#include <armadillo>

typedef struct {
	Eigen::Matrix3f Rx;
	Eigen::Matrix3f Ry;
	Eigen::Matrix3f Rz;
} rotations;

rotations calcrotationmatrix(double rx, double ry, double rz);
Eigen::Matrix3f calchomography(cv::Mat image, double rx, double ry, double rz);
cv::Mat hom3(cv::Mat image, double rx, double ry, double rz);
Eigen::Vector4i calccorners(Eigen::Matrix3f H, int height, int width);

const double PI		= 3.141592653589793;
const double INF	= abs(1.0/0.0);
