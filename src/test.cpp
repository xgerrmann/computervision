#include "test.hpp"
int main(){
	cv::Mat image_in = cv::imread("image.jpg",CV_LOAD_IMAGE_COLOR);
	cv::Mat image_out(image_in.rows, image_in.cols,CV_8UC3);

	copy(image_in, image_out);

	cv::imshow("Input",image_in);
	cv::imshow("Output",image_out);

	cv::waitKey();
	return 0;
}
