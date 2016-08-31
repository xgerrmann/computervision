#include "homography.hpp"
// script to test and perform homographies on an example image
// script based on: http://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog/1886060#1886060

int main(){
	cv::Mat image_in = cv::imread("image.jpg", CV_LOAD_IMAGE_COLOR);
	std::string windowname	= "Original image";
	//cv::namedWindow(windowname);
	
	float	rx = 0.0*PI;
	//float	ry = 0.24*PI;
	float	ry = 0.00*PI;
	float	rz = 0.0*PI;
//	float	rz = 0.0*PI;
	
	Rxyz rot				= calcrotationmatrix(rx, ry, rz);
	trans transformations;
	transformations["tx"]	= 0;
	transformations["ty"]	= 0;
	transformations["tz"]	= 0;
	//transformations["rx"]	= rx;
	//transformations["ry"]	= ry;
	//transformations["rz"]	= rz;
	transformations["rx"]	= -0.020254;
	transformations["ry"]	= -0.012746;
	transformations["rz"]	= -0.0873265;
	Eigen::Matrix3f Rt	= rot.Rz*rot.Ry*rot.Rx;
	int width_screen, height_screen;
	width_screen	= 1920;
	height_screen	= 1080;
	//gputimer watch;
	//watch.start();
	cv::Mat image_out(height_screen,width_screen,CV_8UC3);
	cv::Mat image_out_test(image_in.rows,image_in.cols,CV_8UC3);
	hom(image_in, image_out_test, transformations, width_screen, height_screen);
	//copy(image_in, image_out_test);
	
	cv::imshow("input", image_in);
	cv::imshow("output",image_out_test);
	cv::waitKey();
	//watch.stop();
	//std::cerr << "Finished, only need to show." << std::endl;
//	Show results
	//cv::namedWindow("Hom",cv::WINDOW_OPENGL);
	//cv::imshow("Hom",image_out);
	////cv::imshow("Hom",image_in);
	////cv::imshow("Hom",image);
	//cv::waitKey(0);

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
