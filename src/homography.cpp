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
	//float	rz = 0.5*PI;
	float	rz = 0.5*PI;
	
	cv::Mat im3 = hom(image,rx,ry,rz);

//	Show results
//	cv2.imshow('Hom0',im0)
//	cv2.imshow('Hom1',im1)
//	cv2.imshow('Hom2',im2)
	cv::imshow("Hom3",im3);
	cv::waitKey(0);

// C++
// Looping: Elapsed time 0.004 (rough average)

// Python:
//#	Hom0: Elapsed time 0.00331997871399
//#	Hom1: Elapsed time 2.2098929882
//#	Hom2: Elapsed time 0.0985150337219
//#	Hom3: Elapsed time 0.00763010978699
	return 0;
}
