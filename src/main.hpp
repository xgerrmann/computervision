#include "paths.hpp"

#include "homography.hpp"

#include "../lib/timer/timer.hpp"

// ## dlib was
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/serialize.h>
//#include <dlib/image_io.h>
#include <iostream>

// ## opencv2
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp> // for cast from cv::Mat to Eigen::Matrix
//Attention tracker
#include "../include/attention-tracker/src/head_pose_estimation.hpp"

#define EVER ;;

#define _DEBUG_

class transformation_manager{
	trans transformation_init;
	std::list<std::tuple<timer::tp, trans>> transformation_history;
	timer::tp tstart, tend;		// Start and end time of the currently stored history.
	//timer::dt max_history_length;		// Max history length.
	double max_history_length;		// Max history length.

	public:
		transformation_manager();			// Constructor
		trans add(trans transformation);	// Method to add transformation to the history, returns a transformation based on its history; 
};
// Constructor
transformation_manager::transformation_manager(){
	max_history_length	= double(1.0);			// [s]
	//tstart			= timer::now();	// [s]
	//tend				= timer::now();	// [s]
}
trans transformation_manager::add(trans transformation){
	timer::tp tp_now	= timer::now();
	tend				= tp_now;
	if(transformation_history.empty()){
		tstart				= tp_now;
		transformation_init	= transformation;
	}
	timer::dt dt_his	= tend-tstart;
	//std::cerr << "his:" << max_history_length << std::endl;
	//std::cerr << "dt: " << double(dt_his.count()) << std::endl;
	//std::cerr << "len: "<< transformation_history.size() << std::endl;
	transformation_history.push_front(std::make_tuple(tp_now,transformation));
	//std::cerr << "len: "<< transformation_history.size() << std::endl;
	while((tend-tstart).count()>max_history_length){
		// delete oldest transformation in history
		transformation_history.pop_back();
		// Update tstart with the timestamp of the last element
		tstart	= std::get<0>(transformation_history.back());
	}
	// right now, return the transformation directly
	auto dof_out	= transformation.begin();
	auto dof_init	= transformation_init.begin();
	while(dof_out != transformation.end()){
		//std::cerr << "PRE  dof_in: " << dof_out->second << ", dof_init: " << dof_init->second << std::endl;
		dof_out->second = dof_out->second-dof_init->second;
		//std::cerr << "POST dof_in: " << dof_out->second << ", dof_init: " << dof_init->second << std::endl;
		++dof_out;
		++dof_init;
	}
	return transformation;
}


dlib::full_object_detection detect_face(std::string window_face, std::string window_image, dlib::shape_predictor predictor, dlib::frontal_face_detector detector, cv::Mat frame);
void showshape(std::string window_face, cv::Mat frame, dlib::full_object_detection shape);
void draw_polyline(cv::Mat img,dlib::full_object_detection shape, int start, int stop, bool isClosed = false);
