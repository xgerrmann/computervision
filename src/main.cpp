#include "main.hpp"


//## Sources
//# Example:		http://dlib.net/face_landmark_detection.py.html
//# Speeding up:	http://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
void draw_polyline(cv::Mat img,dlib::full_object_detection shape, int start, int stop, bool isClosed){
//	From: http://www.learnopencv.com/speeding-up-dlib-facial-landmark-detector/
	cv::vector<cv::Point> points;
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

    //auto estimator = HeadPoseEstimation(argv[1]);
    auto estimator = HeadPoseEstimation(trained_model);
	//dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	//dlib::shape_predictor predictor;
	//dlib::deserialize(trained_model) >> predictor;

	cv::Mat image;
	image = cv::imread(default_image);
	std::string window_image = "Image";
	cv::namedWindow("Image");
	cv::imshow("Image",image);
	cv::waitKey(1);
	std::string window_face = "Face";
	cv::namedWindow(window_face);
	cv::VideoCapture video_in(0);
	// adjust for your webcam!
	video_in.set(CV_CAP_PROP_FRAME_WIDTH, 640);
	video_in.set(CV_CAP_PROP_FRAME_HEIGHT, 480);
	estimator.focalLength		= 500;
	estimator.opticalCenterX	= 320;
	estimator.opticalCenterY	= 240;

	if(!video_in.isOpened()){ // Early return if no frame is captured by the cam
		std::cerr << "No frame capured by camera, try running again.";
		return -1;
	}
	timer watch;
	cv::Mat frame;
	double subsample_detection_frame = 2.0;
	for(EVER){
		watch.start();
		video_in >> frame;
		watch.lap("Get frame");
		estimator.update(frame,subsample_detection_frame);
		watch.lap("Update estimator");
		cv::imshow(window_face,frame);
		float rx, ry, rz;
		cv::Mat tmp_cv(4,4,CV_64FC1); // double data type, single channel
		Eigen::Matrix4d tmp_eigen;
		Eigen::Matrix4f pose;
		for(head_pose pose_head : estimator.poses()) {
			watch.start();
			tmp_cv = cv::Mat(pose_head.get_minor<4,4>(0,0));
			cv::cv2eigen(tmp_cv,tmp_eigen);
			pose = tmp_eigen.cast<float>();
			//std::cerr << "Head pose (pose):\n" << pose << std::endl;
			cv::Mat im = hom(image,pose);
			watch.lap("Calculate new image");
			cv::imshow(window_image,im);
			watch.lap("imshow");
		}
		char key = (char)cv::waitKey(1);
		if(key == 27){
			std::cerr << "Program halted by user.";
			return 0;
		}
	}
	// Release webcam
	video_in.release();
	return 0;
}
//		adjustwindow(window_image, image, headpose)

// Attention tracker sample
//#ifdef HEAD_POSE_ESTIMATION_DEBUG
//        imshow("headpose", estimator._debug);
//        if (waitKey(10) >= 0) break;
//#endif

