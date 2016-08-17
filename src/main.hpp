#include "paths.hpp"

#include "homography.hpp"

// ## dlib
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/serialize.h>
//#include <dlib/image_io.h>
#include <iostream>

// ## opencv2
//#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//Attention tracker
//#include "../include/attention-tracker/src/head_pose_estimation.hpp"
//#include "../include/attention-tracker/src/head_pose_estimation.hpp"
//#include "attention-tracker/src/head_pose_estimation.hpp"
//#include "../include/attention-tracker/head_pose_estimation.hpp"
#include "/home/xander/computervision/include/attention-tracker/src/head_pose_estimation.hpp"

//using namespace dlib;
//using namespace std;
//using namespace cv;

dlib::full_object_detection detect_face(std::string window_face, std::string window_image, dlib::shape_predictor predictor, dlib::frontal_face_detector detector, cv::Mat frame);
void showshape(std::string window_face, cv::Mat frame, dlib::full_object_detection shape);
void draw_polyline(cv::Mat img,dlib::full_object_detection shape, int start, int stop, bool isClosed = false);
