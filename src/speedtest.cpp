// speedtest.cpp
// X.G.Gerrmann
//
#include "main.hpp"

int main(){
// Partially based on sample of attention tracker
	int n_trials	= 1000;
	int n_sizes		= 100;
	Eigen::MatrixXi sizes(n_sizes, 2);
	Eigen::MatrixXi times(n_trials, n_sizes);
	std::string window_image = "Image";
	cv::namedWindow(window_image,cv::WINDOW_OPENGL);
	cv::VideoCapture video_in(0);
	cv::RNG cvrand;
	for(int i_size = 1; i_size<=n_sizes; i_size ++){
		int width			= i_size*1920/n_sizes;
		int height			= i_size*1080/n_sizes;
		sizes(i_size,0) = width;
		sizes(i_size,1) = height;

		cv::Mat tmp0			= cv::imread(default_image);
		cv::Mat image_in_tmp	= cv::Mat::ones(height,width,CV_8UC3);
		//cv::randu(image_in_tmp,0,255);
		cvrand.fill(image_in_tmp,cv::RNG::UNIFORM,0,255,true);
		// TODO create different random image
		

		const cv::cuda::GpuMat image_in(image_in_tmp);
		
		int size_buff = 5;
		double fps = video_in.get(CV_CAP_PROP_FPS);
		//std::cerr << "Max framerate: " << fps << std::endl;
		//int width_webcam, height_webcam;
		//width_webcam	= 640;
		//height_webcam	= 480;
		//video_in.set(CV_CAP_PROP_FPS, 30);
		//cv::Mat frame(height_webcam,width_webcam,CV_8UC3);
	
		if(!video_in.isOpened()){ // Early return if no frame is captured by the cam
			std::cerr << "No frame capured by camera, try running again.";
			return -1;
		}
		// Determine size of device main display
		Display* disp = XOpenDisplay(NULL);
		Screen*  scrn = DefaultScreenOfDisplay(disp);
		int height_screen	= scrn->height- 50; // adjust for top menu in ubuntu
		int width_screen	= scrn->width - 64; // adjust for sidebar in ubuntu
		//std::cerr << "Screen size (wxh): "<<width_screen<<", "<<height_screen<<std::endl;
		gputimer watch;
		double subsample_detection_frame = 3.0;
		cv::cuda::GpuMat image_out(height_screen, width_screen, CV_8UC3);
		image_out.setTo(0);
		int n_frames_pose_average = 4;
		transformation_manager trans_mngr(n_frames_pose_average);
		int initial = 1;
		for(int trial = 0; trial<n_trials; trial++){
			//video_in >> frame;
			// Reset im_out (only if head is detected)
			image_out.setTo(0);
			
			trans transformation;
			if(initial == 1){
			//std::cerr << "Rotations:"  << rotations << std::endl;
				transformation.tx = 0;
				transformation.ty = 0;
				transformation.tz = 0;
				transformation.rx = 0;
				transformation.ry = 0;
				transformation.rz = 0;
				initial = 0;
			}else{
				transformation.tx = 0;
				transformation.ty = 0;
				transformation.tz = 0;
				transformation.rx = 0;
				transformation.ry = 0.1*PI;
				transformation.rz = 0.25*PI;
			}
			trans transformation_update  = trans_mngr.add(transformation);
			watch.start();
			hom(image_in, image_out, transformation_update,width_screen,height_screen);
			// Store time
			//times(trial,i_size) = watch.lap(printf("Trial: %i, t", trial));
			std::string text = " ";
			text += std::to_string(trial);
			times(trial,i_size) = watch.lap(text);
			cv::imshow(window_image,image_out);
			char key = (char)cv::waitKey(1);
			if(key == 27){
				std::cerr << "Program halted by user.\n";
				break;
			}
			#if(_MAIN_TIMEIT)
			watch.lap("Imshow");
			#endif
			#if(_MAIN_DEBUG)
			double t_total = watch.stop();
			std::cerr << "Framerate: " << 1/t_total << "[Hz]" << std::endl;
			std::cerr << "#############################################################" << std::endl;
			#endif
		}
	}
	// Store sizes and trials in a .txt file
	std::string dir = "media/results"
	//std::string 
	// Close window
	cv::destroyWindow(window_image);
	// Release webcam
	video_in.release();
	return 0;
}
