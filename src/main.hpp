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
#define _MAIN_DEBUG 1

#define _MAIN_TIMEIT 1
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
	std::cerr << "max history length [s]:     " << max_history_length.count() << std::endl;
	std::cerr << "current history length [s]: " << double(dt_his.count()) << std::endl;
	std::cerr << "history length (frames):    " << transformation_history.size() << std::endl;
	// Add the new transformation to the history list
	transformation_history.push_front(std::make_tuple(tp_now,transformation));
	std::cerr << "len: "<< transformation_history.size() << std::endl;
	std::cerr << "tend-tstart: "<< float((tend-tstart).count()) << std::endl;
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
	
	std::cerr << "n_avg: " << n_avg_max << std::endl;
	std::cerr << "size: " << transformation_history.size() << std::endl;
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
