#include <iostream>

//#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "paths.hpp"
#include <math.h>
// ## Eigen
#include <Eigen/Dense>

typedef struct {
	Eigen::Matrix3f Rx;
	Eigen::Matrix3f Ry;
	Eigen::Matrix3f Rz;
} rotations;

rotations calcrotationmatrix(double rx, double ry, double rz);
int calchomography(cv::Mat image, double rx, double ry, double rz);
cv::Mat hom3(cv::Mat image, double rx, double ry, double rz);

const double PI = 3.141592653589793;

