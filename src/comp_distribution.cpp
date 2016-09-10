// comp_distribution.cpp
// X.G.Gerrmann
//
#include "main.hpp"

int main(){
// Partially based on sample of attention tracker
	int n_trials	= 1000;
	int n_sections	= 7;
	Eigen::MatrixXf times(n_trials, n_sections);

    auto estimator = HeadPoseEstimation(trained_model);

	cv::Mat image_in_tmp	= cv::imread(default_image);
	const cv::cuda::GpuMat image_in(image_in_tmp);
	std::string window_image = "Image";
	// 
	cv::namedWindow(window_image,cv::WINDOW_OPENGL);
	
	cv::VideoCapture video_in(0);
	int size_buff = 6;
	double fps = video_in.get(CV_CAP_PROP_FPS);
	std::cerr << "Max framerate: " << fps << std::endl;
	int width_webcam, height_webcam;
	// TODO: get width and height from webcam instead of hardcoding
	width_webcam	= 640;
	height_webcam	= 480;
	video_in.set(CV_CAP_PROP_FPS, 30);
	//video_in.set(CV_CAP_PROP_FRAME_WIDTH, width_webcam);
	//video_in.set(CV_CAP_PROP_FRAME_HEIGHT, height_webcam);
	estimator.focalLength		= 500;
	estimator.opticalCenterX	= 320;
	estimator.opticalCenterY	= 240;
	cv::Mat frame(height_webcam,width_webcam,CV_8UC3);

	if(!video_in.isOpened()){ // Early return if no frame is captured by the cam
		std::cerr << "No frame capured by camera, try running again.";
		return -1;
	}
	// Determine size of device main display
	Display* disp = XOpenDisplay(NULL);
	Screen*  scrn = DefaultScreenOfDisplay(disp);
	int height_screen	= scrn->height- 50; // adjust for top menu in ubuntu
	int width_screen	= scrn->width - 64; // adjust for sidebar in ubuntu
	std::cerr << "Screen size (wxh): "<<width_screen<<", "<<height_screen<<std::endl;
	gputimer watch;
	double subsample_detection_frame = 3.0;
	cv::cuda::GpuMat image_out(height_screen, width_screen, CV_8UC3);
	image_out.setTo(0);
	int n_frames_pose_average = 4;
	transformation_manager trans_mngr(n_frames_pose_average);
	for(int i =0; i<n_trials; i++){
		std::cerr << "Trial: " << i << std::endl;
		watch.start();
		
		video_in >> frame;
		times(i,0) = watch.lap(); // get frame from webcam
		estimator.update(frame,subsample_detection_frame);
		times(i,1) = watch.lap(); // estimate head pose
		
		// Reset im_out (only if head is detected)
		image_out.setTo(0);
		
		times(i,2) = watch.lap(); // reset image

		//std::cerr << "Rotations:"  << rotations << std::endl;
		trans transformation;
		transformation.tx = 0;
		transformation.ty = 0;
		transformation.tz = 0;
		transformation.rx = 0;
		transformation.ry = 0;
		transformation.rz = 0;

		trans transformation_update  = trans_mngr.add(transformation);
		times(i,3) = watch.lap();// manage transformations
		hom(image_in, image_out, transformation_update,width_screen,height_screen);
		times(i,4) = watch.lap(); // calc new image
		
		cv::imshow(window_image,image_out);
		char key = (char)cv::waitKey(1);
		if(key == 27){
			std::cerr << "Program halted by user.\n";
			break;
		}
		times(i,5) = watch.lap(); // show image
		times(i,6) = watch.stop();// total time
	}
	std::cerr << "Finished tests" << std::endl;
	// Store sizes and trials in a .txt file
	std::string dir		= "media/results/";
	std::string f_times	= "comp_distribution.csv";
	const static Eigen::IOFormat CSVFormat(6, true, ", ", "\n");
	std::cerr << dir+f_times << std::endl;
	std::ofstream file_comp(dir+f_times);
	std::cerr << "Start Writing" << std::endl;
	if(file_comp.is_open()){
		std::cerr << "file_times is opened" << std::endl;
		file_comp << times.format(CSVFormat);
		file_comp.close();
	}
	std::cerr << "Finished writing" << std::endl;

	// Close window
	cv::destroyWindow(window_image);
	// Release webcam
	std::cerr << "release webcam" << std::endl;
	video_in.release();
	std::cerr << "return" << std::endl;
	return 0;
}
