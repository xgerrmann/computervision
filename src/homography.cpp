#include "homography.hpp"
// script to test and perform homographies on an example image
// script based on: http://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog/1886060#1886060

int main(){
	cv::Mat image;
	image = cv::imread(default_image);
	std::string windowname	= "Original image";
	cv::namedWindow(windowname);
	cv::imshow(windowname,image);
	cv::waitKey(100);
	
	float	rx = 0.0*PI;
	float	ry = 0.0*PI;
	float	rz = 0.5*PI;
	
	Rxyz rot				= calcrotationmatrix(rx, ry, rz);
	trans transformations;
	transformations["dx"]	= 0;
	transformations["dy"]	= 0;
	transformations["dz"]	= 0;
	transformations["rx"]	= rx;
	transformations["ry"]	= ry;
	transformations["rz"]	= rx;
	Eigen::Matrix3f Rt	= rot.Rz*rot.Ry*rot.Rx;
	int wmax, hmax;
	wmax = 1920;
	hmax = 1080;
	timer watch;
	watch.start();
	cv::Mat im			= hom(image,transformations,wmax,hmax);
	watch.stop();

//	Show results
	cv::imshow("Hom",im);
	cv::waitKey(0);

// C++
// Looping: Elapsed time 0.004 (rough average)
// Looping: Elapsed time 0.03 (rough average) small 250x250 image
// Looping: Elapsed time 0.06 (rough average) Large image

// Python:
//#	Hom0: Elapsed time 0.00331997871399
//#	Hom1: Elapsed time 2.2098929882
//#	Hom2: Elapsed time 0.0985150337219
//#	Hom3: Elapsed time 0.00763010978699
	return 0;
}
