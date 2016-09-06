#include "paths.hpp"

#include "homography.hpp"

// ##(CPU) timer
#include "../lib/timer/timer.hpp"

// ## gputimer
#include "../lib/gputimer/gputimer.hpp"

// ## dlib was
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/serialize.h>
//#include <dlib/image_io.h>
#include <iostream>
#include <algorithm>

// ## opencv2
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp> // for cast from cv::Mat to Eigen::Matrix
#include <opencv2/core/cuda.hpp>

//Attention tracker
#include "../include/attention-tracker/src/head_pose_estimation.hpp"

#define EVER ;;

#ifdef _DEBUG_
#define _MAIN_DEBUG 1
#endif
//#define _MAIN_DEBUG 1

//#define _MAIN_TIMEIT 1
//#define _MAIN_TIMEIT 1

class transformation_manager{
	trans transformation_init;
	int n_average; // Number of frames to average the headpose over
	std::list<std::tuple<timer::tp, trans>> transformation_history;
	timer::tp tstart, tend;		// Start and end time of the currently stored history.
	//timer::dt max_history_length;		// Max history length.
	timer::dt max_history_length;		// Max history length.

	public:
		transformation_manager(int frames_average);			// Constructor
		trans add(trans transformation);	// Method to add transformation to the history, returns a transformation based on its history; 
};
// Constructor
transformation_manager::transformation_manager(int frames_average){
	max_history_length	= std::chrono::seconds(1);	// [s]
	//tstart			= timer::now();	// [s]
	//tend				= timer::now();	// [s]
	n_average = frames_average;
}
trans transformation_manager::add(trans transformation){
	timer::tp tp_now	= timer::now();
	tend				= tp_now;
	if(transformation_history.empty()){
		tstart				= tp_now;
		transformation_init	= transformation;
	}
	timer::dt dt_his	= tend-tstart; // This is the length (in seconds) of the currently stored history
	#ifdef _MAIN_DEBUG_
	std::cerr << "max history length [s]:     " << max_history_length.count() << std::endl;
	std::cerr << "current history length [s]: " << double(dt_his.count()) << std::endl;
	std::cerr << "history length (frames):    " << transformation_history.size() << std::endl;
	#endif
	// Add the new transformation to the history list
	transformation_history.push_front(std::make_tuple(tp_now,transformation));
	#ifdef _MAIN_DEBUG_
	std::cerr << "len: "<< transformation_history.size() << std::endl;
	std::cerr << "tend-tstart: "<< float((tend-tstart).count()) << std::endl;
	#endif
	while(tend-tstart>max_history_length){
		// delete oldest transformation in history
		transformation_history.pop_back();
		// Update tstart with the timestamp of the last element
		tstart	= std::get<0>(transformation_history.back());
	}
	// right now, return the transformation directly
	// For every element in the transformation: tx, ty, tz, rx, ry and rz substract the initial pose
	trans trans_new;
	trans_new.tx = 0;
	trans_new.ty = 0;
	trans_new.tz = 0;
	trans_new.rx = 0;
	trans_new.ry = 0;
	trans_new.rz = 0;
	auto trans_tup_tmp	= transformation_history.begin();
	int n_avg_max		= std::min(n_average,int(transformation_history.size()));
	
	#ifdef _MAIN_DEBUG_
	std::cerr << "n_avg: " << n_avg_max << std::endl;
	std::cerr << "size: " << transformation_history.size() << std::endl;
	#endif
	for(int i =0; i<n_avg_max; i++){
		trans trans_tmp = (std::get<1>(*trans_tup_tmp)); // first element of tuple is time, second is a transformation
		trans_new.tx += (trans_tmp.tx-transformation_init.tx)/n_avg_max;
		trans_new.ty += (trans_tmp.ty-transformation_init.ty)/n_avg_max;
		trans_new.tz += (trans_tmp.tz-transformation_init.tz)/n_avg_max;
		trans_new.rx += (trans_tmp.rx-transformation_init.rx)/n_avg_max;
		trans_new.ry += (trans_tmp.ry-transformation_init.ry)/n_avg_max;
		trans_new.rz += (trans_tmp.rz-transformation_init.rz)/n_avg_max;
		++trans_tup_tmp; // makes sure the first is also skipped
	}
	return trans_new;
}


dlib::full_object_detection detect_face(std::string window_face, std::string window_image, dlib::shape_predictor predictor, dlib::frontal_face_detector detector, cv::Mat frame);
void showshape(std::string window_face, cv::Mat frame, dlib::full_object_detection shape);
void draw_polyline(cv::Mat img,dlib::full_object_detection shape, int start, int stop, bool isClosed = false);

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
