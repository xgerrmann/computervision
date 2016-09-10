// main.cpp
// X.G.Gerrmann
//
#include "main.hpp"


int main(int argc, char* argv[]){
// Partially based on sample of attention tracker
	float arg_tx,arg_ty,arg_tz,arg_rx,arg_ry,arg_rz;
	bool manual = false;
	if(argc>0){
		if(not (argc == 7)){
			std::cerr << "Usage: \n./screenshots tx ty tz rx ry rz \n or just:\n./screenshots" <<std::endl;
			exit(0);
		}
		arg_tx = float(atof(argv[1]));
		arg_ty = float(atof(argv[2]));
		arg_tz = float(atof(argv[3]));
		arg_rx = float(atof(argv[4]));
		arg_ry = float(atof(argv[5]));
		arg_rz = float(atof(argv[6]));
		manual = true;
	}
	auto estimator = HeadPoseEstimation(trained_model);

	cv::Mat image_in_tmp	= cv::imread(default_image);
	const cv::cuda::GpuMat image_in(image_in_tmp);
	std::string window_image = "Image";
	cv::namedWindow(window_image,cv::WINDOW_OPENGL);
	
	cv::VideoCapture video_in(0);
	int size_buff = 5;
	double fps = video_in.get(CV_CAP_PROP_FPS);
	int width_webcam, height_webcam;
	width_webcam	= 640;
	height_webcam	= 480;
	video_in.set(CV_CAP_PROP_FPS, 30);
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
	gputimer watch;
	double subsample_detection_frame = 3.0;
	//cv::Mat im_out(height_screen,width_screen,CV_8UC3);
	cv::cuda::GpuMat image_out(height_screen, width_screen, CV_8UC3);
	image_out.setTo(0);
	int n_frames_pose_average = 4;
	transformation_manager trans_mngr(n_frames_pose_average);
	int i_frame = 0;
	for(EVER){
		//#if _MAIN_DEBUG || _MAIN_TIMEIT
		watch.start();
		//#endif
		video_in >> frame;
		#if(_MAIN_TIMEIT)
		watch.lap("Get frame");
		#endif
		estimator.update(frame,subsample_detection_frame);
		#if(_MAIN_TIMEIT)
		watch.lap("Update estimator");
		#endif
		// TODO: does not have to be a for loop..
		for(head_pose pose_head : estimator.poses()) {
			// Reset im_out (only if head is detected)
			image_out.setTo(0);
			
			cv::Mat rotations		= pose_head.rvec;
			cv::Mat translations	= pose_head.tvec;

			//std::cerr << "Rotations:"  << rotations << std::endl;
			trans transformation;
			trans transformation_update  = trans_mngr.add(transformation);
			if(manual==false){
				transformation.tx = (float)translations.at<double>(0);
				transformation.ty = (float)translations.at<double>(1);
				transformation.tz = (float)translations.at<double>(2);
				transformation.rx = (float)rotations.at<double>(0);
				transformation.ry = (float)rotations.at<double>(1);
				transformation.rz = (float)rotations.at<double>(2);
				transformation_update  = trans_mngr.add(transformation);
			}else{
				transformation.tx = arg_tx;
				transformation.ty = arg_ty;
				transformation.tz = arg_tz;
				transformation.rx = arg_rx*PI;
				transformation.ry = arg_ry*PI;
				transformation.rz = arg_rz*PI;
				transformation_update = transformation;
			}
			#if(_MAIN_TIMEIT)
			watch.lap("Manage transformations");
			#endif
			hom(image_in, image_out, transformation_update,width_screen,height_screen);
			#if(_MAIN_TIMEIT)
			watch.lap("Calculate new image");
			#endif
			//// draw feature points on face
			//draw_polyline(frame,pose_head, 0, 40, false){
		}
		if(not manual){	
		// Place webcam image in frame
			cv::Rect slice	= cv::Rect(width_screen-width_webcam,height_screen-height_webcam,width_webcam, height_webcam);
			cv::cuda::GpuMat frame_gpu(estimator._debug);
			frame_gpu.copyTo(image_out(slice));
		}
		// Show image
		//cv::imshow(window_image,image_out);
		cv::Mat im_out_tmp;
		image_out.download(im_out_tmp);
		if(i_frame%10 == 0 and not manual){
			cv::imwrite( "media/screenshots/"+std::to_string(i_frame)+"_"".png", im_out_tmp);
		}
		else if(manual==true){
			std::cerr << "yes" << std::endl;
			cv::imwrite( "media/screenshots/"	+std::to_string(arg_tx)+"_"
												+std::to_string(arg_ty)+"_"
												+std::to_string(arg_tz)+"_"
												+std::to_string(arg_rx)+"_"
												+std::to_string(arg_ry)+"_"
												+std::to_string(arg_rz)+".png", im_out_tmp);
		}
		//cv::imshow("debug",estimator._debug);
		char key = (char)cv::waitKey(1);
		if(key == 27){
			std::cerr << "Program halted by user.\n";
			break;
		}
		
		// store screenshots
		#if(_MAIN_TIMEIT)
		watch.lap("Imshow");
		#endif
		#if(_MAIN_DEBUG)
		double t_total = watch.stop();
		std::cerr << "Framerate: " << 1/t_total << "[Hz]" << std::endl;
		std::cerr << "#############################################################" << std::endl;
		#endif
		i_frame ++;
		
		if(manual){// only do one loop
			break;
		}
	}

	// Close window
	cv::destroyWindow(window_image);
	// Release webcam
	video_in.release();
	return 0;
}
