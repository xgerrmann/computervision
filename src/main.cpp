#include "main.hpp"


//## Sources
//# Example:		http://dlib.net/face_landmark_detection.py.html
//# Speeding up:	http://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
void draw_polyline(cv::Mat img,dlib::full_object_detection shape, int start, int stop, bool isClosed){
//	From: http://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
	std::vector<cv::Point> points;
	for(int i=start; i<stop; i++){
		points.push_back(cv::Point(shape.part(i).x(), shape.part(i).y()));
	}
	cv::polylines(img, points, isClosed, cv::Scalar(255,0,0), 1, 16);
}

void showshape(std::string window_face, cv::Mat frame, dlib::full_object_detection shape){
//	Based on: http://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
	draw_polyline(frame, shape, 0,  16, false);		// Jaw line
	draw_polyline(frame, shape, 17, 21, false);		// Left eyebrow
	draw_polyline(frame, shape, 22, 26, false);		// Right eyebrow
	draw_polyline(frame, shape, 27, 31, false);		// Nose bridge
	draw_polyline(frame, shape, 30, 35, true);		// Lower nose
	draw_polyline(frame, shape, 36, 41, true);		// Left eye
	draw_polyline(frame, shape, 42, 47, true);		// Right Eye
	draw_polyline(frame, shape, 48, 59, true);		// Outer lip
	draw_polyline(frame, shape, 60, 67, true);		// Inner lipt
	// lines are drawn in the image, image is passed by reference
}

dlib::full_object_detection detect_face(std::string window_face, std::string window_image, dlib::shape_predictor predictor, dlib::frontal_face_detector detector, cv::Mat frame){
//	Ask the detector to find the bounding boxes of each face. The 1 in the
//	second argument indicates that we should upsample the image 1 time. This
//	will make everything bigger and allow us to detect more faces.
//	Detect face
	// TODO: time detection
	double subsample = 1.0/2.0;
	cv::Mat frame_sub;
	cv::resize(frame,frame_sub,cv::Size(0,0),subsample,subsample);

	dlib::cv_image<dlib::bgr_pixel> cimg(frame_sub); // convert cv::Mat to something dlib can work with
	std::vector<dlib::rectangle> dets = detector(cimg);
	// Print detection time
	if(dets.size()==0) // Early return if no face is detected
		throw "No face detected";
		//return NULL;
//	print("Number of faces detected: {}".format(len(dets)))
//	# TODO: use only face with highest detection strength: other faces should be ignored
//	# TODO: make a model of the location of the face for faster detection

//	Rescale detected recangle in subsampled image
	double left		= dets.at(0).left()/subsample;
	double top		= dets.at(0).top()/subsample;
	double right	= dets.at(0).right()/subsample;
	double bottom	= dets.at(0).bottom()/subsample;
	
	std::cerr << "Left: " << left << ", Top: " << top << ", Right: " << right << ", Bottom: " << bottom << "\n";
	dlib::rectangle d = dlib::rectangle(left,top,right,bottom);
	
//	Get the landmarks/parts for the face in box d.
	dlib::cv_image<dlib::bgr_pixel> tmpimg(frame); // convert cv::Mat to something dlib can work with
	dlib::full_object_detection shape = predictor(tmpimg,d);

//	Draw the face landmarks on the screen.
	showshape(window_face,frame,shape);

	return shape;
}

int main(){
// Partially based on sample of attention tracker

    auto estimator = HeadPoseEstimation(trained_model);

	cv::Mat image;
	image = cv::imread(default_image);
	cv::cuda::GpuMat image_in(image);
	std::string window_image = "Image";
	//cv::namedWindow(window_image,cv::WINDOW_OPENGL);
	cv::namedWindow(window_image,cv::WINDOW_OPENGL);
	
	cv::VideoCapture video_in(0);
	int width_webcam, height_webcam;
	// TODO: get width and height from webcam instead of hardcoding
	width_webcam	= 640;
	height_webcam	= 480;
	video_in.set(CV_CAP_PROP_FPS, 15);
	video_in.set(CV_CAP_PROP_FRAME_WIDTH, width_webcam);
	video_in.set(CV_CAP_PROP_FRAME_HEIGHT, height_webcam);
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
	//cv::Mat im_out(height_screen,width_screen,CV_8UC3);
	cv::cuda::GpuMat image_out(height_screen,width_screen,CV_8UC3);
	cv::Mat tmp_cv(4,4,CV_64FC1); // double data type, single channel
	Eigen::Matrix4d tmp_eigen;
	Eigen::Matrix4f pose;
	transformation_manager trans_mngr;
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
		// Reset im_out
		image_out.setTo(cv::Scalar(0));
		for(head_pose pose_head : estimator.poses()) {
			cv::Mat rotations = pose_head.rvec;

			//std::cerr << "Rotations:"  << rotations << std::endl;
			trans transformation;
			transformation["tx"] = 0;
			transformation["ty"] = 0;
			transformation["tz"] = 0;
			transformation["rx"] = (double)rotations.at<double>(0);
			transformation["ry"] = (double)rotations.at<double>(1);
			transformation["rz"] = (double)rotations.at<double>(2);
			//std::cerr << "Trans:" << std::endl;
			//for(auto transform : transformation){
			//	std::cerr << "\t" << std::get<1>(transform) << std::endl;
			//}
			trans transformation_update  = trans_mngr.add(transformation);
			#if(_MAIN_TIMEIT)
			watch.lap("Manage transformations");
			#endif
			//std::cerr << "Trans:";
			//for(auto transform : transformation_update){
			//	std::cerr << "\t" << std::get<1>(transform);
			//}
			//std::cerr << std::endl;
			//watch.lap("Print transformation");
			//im_out = hom(image,transformation_update,width_screen,height_screen);
			// TODO: im_out must be gpuarray
			hom(&image_out, &image_in,transformation_update,width_screen,height_screen);
			#if(_MAIN_TIMEIT)
			watch.lap("Calculate new image");
			#endif
		}
		#if(_MAIN_DEBUG)
			cv::Rect slice	= cv::Rect(width_screen-width_webcam,height_screen-height_webcam,width_webcam, height_webcam);
			frame.copyTo(image_out(slice));
		#endif
		cv::imshow(window_image,image_out);
		char key = (char)cv::waitKey(1);
		if(key == 27){
			std::cerr << "Program halted by user.\n";
			break;
		}
		#if(_MAIN_TIMEIT)
		watch.lap("Imshow");
		#endif
		//#if(_MAIN_DEBUG)
		double t_total = watch.stop();
		std::cerr << "Framerate: " << 1/t_total << "[Hz]" << std::endl;
		//#endif

	}
	// Close window
	cv::destroyWindow(window_image);
	// Release webcam
	video_in.release();
	return 0;
}
