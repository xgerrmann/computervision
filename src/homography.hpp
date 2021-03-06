// homography.hpp
#ifndef homography_h
#define homography_h

#include <iostream>     // std::cout
//#include <algorithm>    // std::max

// (CPU) timer
//#include "../lib/timer/timer.hpp"

// ## GPUtimer
#include "../lib/gputimer/gputimer.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

//#include <opencv2/imgproc/imgproc.hpp>
#include "paths.hpp"
#include <math.h>
// ## Eigen
//#include <Eigen/Dense>
#include <eigen3/Eigen/Dense> // << changes
//#include <limits.h> // for max values of datatypes
#include <float.h> // for max values of datatypes
// ## Attention tracker
#include "../include/attention-tracker/src/head_pose_estimation.hpp"
// ## Xlib
#include <X11/Xlib.h> // To determine the display size
// ## cudafuncs
#include "cudafuncs.hpp"


#define EVER ;;

//#define _HOM_TIMEIT	0
//#define _HOM_TIMEIT	1

#ifdef _DEBUG_
#define _HOM_DEBUG	1
#endif
//#define _HOM_DEBUG	1

typedef struct {
	Eigen::Matrix3f Rx;
	Eigen::Matrix3f Ry;
	Eigen::Matrix3f Rz;
} Rxyz;

typedef struct trans{
	float tx, ty, tz, rx, ry, rz;
} trans;

void hom(const cv::cuda::GpuMat& image_input, cv::cuda::GpuMat& image_output, trans& transformations, int& width_max, int& height_max);
Rxyz calcrotationmatrix(double rx, double ry, double rz);
//std::vector<float> calcrotations(Eigen::Matrix3f Rt);
Eigen::Matrix3f calchomography(int width, int height, trans transformations);
Eigen::Vector4f box_out(Eigen::Matrix3f H, int height, int width);

const double PI		= 3.141592653589793;
const double INF	= fabs(1.0/0.0); // fabs is absolute function for floating point variables
const float PITCH	= 0.2625*(std::pow(10.0,-3.0)); // [m] pixel pitch (pixel size) assume square pixels, which is generally true
// TODO: pitch is different per device

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// script to test and perform homographies on an example image
// script based on: http://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog/1886060#1886060

Rxyz calcrotationmatrix(float rx, float ry, float rz){
//	source: http://nghiaho.com/?page_id=846
//	source: https://en.wikipedia.org/wiki/3D_projection (uses negative angles?)
//	Results of rotation matrices are consistent with python script
//	std::cerr << "Calcrotationmatrix (input): rx, ry, rz: "<< rx << ", "<< ry << ", "<< rz << std::endl;
	Eigen::Matrix3f Rx;
	Rx <<	1.0,		0.0,		0.0,
			0.0,		cos(rx),	-sin(rx),
			0.0,		sin(rx),	cos(rx);
	Eigen::Matrix3f	Ry;
	Ry <<	cos(ry),	0.0,		sin(ry),
			0.0,		1.0,		0.0,
			-sin(ry),	0.0,		cos(ry);
	Eigen::Matrix3f Rz;
	Rz <<	cos(rz),	-sin(rz),	0.0,
			sin(rz),	cos(rz),	0.0,
			0.0,		0.0,		1.0;
//	std::cerr<<"Rz:\n"<<Rz<<std::endl;
//	std::cerr << "Rx:\n"<< Rx<<"\n";
//	std::cerr << "Ry:\n"<< Ry<<"\n";
//	std::cerr << "Rz:\n"<< Rz<<"\n";
	Rxyz rot;
	rot.Rx=Rx;
	rot.Ry=Ry;
	rot.Rz=Rz;
	return rot;
}

Eigen::Vector4f box_out(Eigen::Matrix3f H, int width_original, int height_original){
// This function calculates the locations in the x,y,z, coordinate system of the resulting
// corner points. From this it determines the width and height of the outgoing image.

// Define corners and set origin to center of image
	float wx, hy;
	wx	= float(width_original-1);	// [pixels]
	hy	= float(height_original-1);	// [pixels]
	Eigen::Vector3f c0, c1, c2, c3;
	c0	<< -wx/2,+hy/2,1;// location of corner 0 <x,y,w>
	c1	<< +wx/2,+hy/2,1;// location of corner 1 <x,y,w>
	c2	<< +wx/2,-hy/2,1;// location of corner 2 <x,y,w>
	c3	<< -wx/2,-hy/2,1;// location of corner 3 <x,y,w>
	std::vector<Eigen::Vector3f> corners;
	corners.push_back(c0);
	corners.push_back(c1);
	corners.push_back(c2);
	corners.push_back(c3);
	
	std::vector<Eigen::Vector2f> corners_proj;
	float x,y,w,scale;
	for(int i = 0; i<4; i++){
		Eigen::Vector3f corner_original, corner_new;
		corner_original = corners.at(i);
		//std::cerr << "Corner_original:\n" << corner_original << "\n";
		corner_new = H*corner_original; // Perform a forward homography on the corner point.
		//std::cerr << "Corner_new:\n" << corner_new << "\n";
		scale = corner_new(2);
		//std::cerr << "Scale: " << scale << std::endl;
		corner_new = corner_new/scale; // !! divide by scale
		//std::cerr << "Corner_new:\n" << corner_new << "\n";
		//std::cerr << "Corner_new (transposed):\n" << corner_new << "\n";
		corners_proj.push_back(corner_new.block<2,1>(0,0));
	}
	
//	Projected corners are in pixels
//	#print 'Projected corners:\n',corners_proj
	float xmin_out, ymin_out, xmax_out, ymax_out, xtmp, ytmp;
	xmin_out	= FLT_MAX;
	ymin_out	= FLT_MAX;
	xmax_out	= FLT_MIN;
	ymax_out	= FLT_MIN;
	for(int i = 0; i<4; i++){
		xtmp = corners_proj.at(i)[0];
		ytmp = corners_proj.at(i)[1];
		if(xtmp < xmin_out)
			xmin_out = xtmp;
		if(xtmp > xmax_out)
			xmax_out = xtmp;
		if(ytmp < ymin_out)
			ymin_out = ytmp;
		if(ytmp > ymax_out)
			ymax_out = ytmp;
	}
//	std::cerr << "xmin: \t" << xmin_out << "\n";
//	std::cerr << "xmax: \t" << xmax_out << "\n";
//	std::cerr << "ymin: \t" << ymin_out << "\n";
//	std::cerr << "ymax: \t" << ymax_out << "\n";
	float height_image_out, width_image_out;
	height_image_out	= ymax_out - ymin_out + 1; // height_image_out is in pixels so +1
	width_image_out		= xmax_out - xmin_out + 1; // width_image_out  is in pixels so +1
//	std::cerr << "Height: " << height_image_out << "\n";
//	std::cerr << "Width:  " << width_image_out << "\n";
//	std::cerr << "xmin:   " << xmin_out << "\n";
//	std::cerr << "ymin:   " << ymin_out << "\n";
	Eigen::Vector4f rectangle;
	rectangle << xmin_out,ymin_out,width_image_out,height_image_out;
	return rectangle; // xmin, ymin, width, height
}

Eigen::Matrix3f calchomography(int width, int height, trans transformations){
// calchomograpphy
// This function calculates the homography matrix, given the rotations rx,ry,rz.
// Coordinate system:
// ^ y+
// |
// --->x+
	#if(_HOM_DEBUG) || (_HOM_TIMEIT)
	std::cerr << "### calchomography <start> ###" << std::endl;
	gputimer watch;
	watch.start();
	#endif

	// Width and height of an image must be > 0
	assert(width>0&&height>0);

	//Rt = Rz*Ry*Rx
	Rxyz rot			= calcrotationmatrix(transformations.rx, transformations.ry, transformations.rz);
	Eigen::Matrix3f Rt	= rot.Rz*rot.Ry*rot.Rx;
	#if(_HOM_DEBUG)
	std::cerr << "tx: " << transformations.tx << std::endl;
	std::cerr << "ty: " << transformations.ty << std::endl;
	std::cerr << "tz: " << transformations.tz << std::endl;
	std::cerr << "rx: " << transformations.rx << std::endl;
	std::cerr << "ry: " << transformations.ry << std::endl;
	std::cerr << "rz: " << transformations.rz << std::endl;
	std::cerr << "Rt:\n"	<< Rt	<< std::endl;
	#endif
	Eigen::Matrix3f Rti	= Rt.inverse();
//	std::cerr << "Rti:\n"	<< Rti	<< std::endl;
//	# define 3 points on the virtual image plane
	Eigen::Vector3f p0, p1, pr0, pr1;
	p0 << 1.0, 0.0, 0.0;	// [m]
	p1 << 0.0, 1.0, 0.0;	// [m]
	//std::cerr << p0 << "\n";
	//std::cerr << p1 << "\n";
	//std::cerr << p2 << "\n";
//	# preform rotation of points
	pr0 = Rt*p0;
	pr1 = Rt*p1;
//	The vectors pr0 and pr1 define the plane.

//	Construct the vectors for the view-lines from the optical center to the corners of the virtual image:
//	Corner numbering:
//	0-------1
//	|		|
//	3-------2
//	pixels
//	First define the corners [x,y]
//	# meters
	float	f, wx, hy;
	f = 0.3;	// [m] Distance of viewer to the screen
	Eigen::Vector3f	e;
	e << 0.0,0.0,f;			// position of viewer (user)
//	#print e
	wx	= float(width-1);	// [pixels]
	hy	= float(height-1);	// [pixels]
	Eigen::Vector3f cp0, cp1, cp2, cp3;
	cp0	<< -wx/2,+hy/2,0;// location of corner 0 <x,y,z> [pixels]
	cp1	<< +wx/2,+hy/2,0;// location of corner 1 <x,y,z> [pixels]
	cp2	<< +wx/2,-hy/2,0;// location of corner 2 <x,y,z> [pixels]
	cp3	<< -wx/2,-hy/2,0;// location of corner 3 <x,y,z> [pixels]
	
	std::vector<Eigen::Vector3f> corners;
	corners.push_back(cp0);
	corners.push_back(cp1);
	corners.push_back(cp2);
	corners.push_back(cp3);
	// make vectors (and convert to meters)
	Eigen::Vector3f c0, c1, c2, c3;
	c0 = PITCH*cp0-e;// vector from eye to corner 0 [m]
	c1 = PITCH*cp1-e;// vector from eye to corner 0 [m]
	c2 = PITCH*cp2-e;// vector from eye to corner 0 [m]
	c3 = PITCH*cp3-e;// vector from eye to corner 0 [m]
	std::vector<Eigen::Vector3f> cornerlines;
	cornerlines.push_back(c0);
	cornerlines.push_back(c1);
	cornerlines.push_back(c2);
	cornerlines.push_back(c3);

//	#print 'Lines:\n',cornerlines
//	# For each intersection a linear combination of the vectors spanning the plane exists
//	# when this combination is found, the exact location of the intersection is known
	std::vector<Eigen::Vector2f> corners_proj;
	for(int i=0; i<4; i++){
		//Find the projection of each corner point on the plane
		//note: origin is still center of the plane
		Eigen::Matrix3f A, Ai;
		Eigen::Matrix<float,3,2> tmp_mat;
		Eigen::Vector3f comb, intersection, coords;
		Eigen::Vector2f tmp_vec, corner;
		//std::cerr << "Corner: " << cornerlines.at(i) << "\n";
		A	<< cornerlines.at(i) , -pr0, -pr1;
		//std::cerr << "A: \n" << A <<"\n";
		Ai		= A.inverse();
		//std::cerr << "Ai: \n" << Ai <<"\n";
		//std::cerr << "e\n" << e <<"\n";
		comb	= Ai*(-e);
		//std::cerr << "Comb:\n" << comb << "\n";
		// Convert real coordinates into pixel coordinates
		float x, y;
		x = comb[1]/PITCH;
		y = comb[2]/PITCH;
		//std::cerr << "x: " << x << "\n";
		//std::cerr << "y: " << y << "\n";
		corner << x,y;
		corners_proj.push_back(corner);
	}

//	The pixel locations of the corners in the projection are now known.
//	calculate the homography
//	source: http://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
	Eigen::Matrix<float,8,8> M1;
	Eigen::Matrix<float,8,1> M2;
	float xA, xB, yA, yB;
	for(int i = 0; i<4; i++){
		xA	= corners.at(i)[0];			// [pixels] original corner
		yA	= corners.at(i)[1];			// [pixels] original corner
		xB	= corners_proj.at(i)[0];	// [pixels] new corner (projected)
		yB	= corners_proj.at(i)[1];	// [pixels] new corner (projected)
		//std::cerr << "xA: " << xA <<"\t-> xB: " << xB << "\n";
		//std::cerr << "yA: " << yA <<"\t-> yB: " << yB << "\n";
		M1.row(i*2)		<< xA,yA,1,0,0,0,-xA*xB,-yA*xB;
		M1.row(i*2+1)	<< 0,0,0,xA,yA,1,-xA*yB,-yA*yB;
		M2.row(i*2)		<< xB;
		M2.row(i*2+1)	<< yB;
	}
	//std::cerr << "M1:\n" << M1 <<"\n";
	//std::cerr << "M2:\n" << M2 <<"\n";
	Eigen::Matrix<float,8,1> Htmp; // column vector with 8 elements
	Eigen::Matrix3f H;
	Htmp = (M1.transpose()*M1).inverse()*(M1.transpose()*M2);
	H << Htmp; // Fill H with Htmp. First down, then right. Therefore needs to be transposed later.
	H.block<1,1>(2,2) << 1.0;
	H.transposeInPlace(); // in place transposition to avoid aliasing
	//std::cerr << "H:\n" << H << "\n";
	// H is calculated correct and results correspond with python script
	#if(_HOM_DEBUG) || (_HOM_TIMEIT)
	watch.stop();
	std::cerr << "### calchomography <end> ###" << std::endl;
	#endif
	return H;
}

void hom(const cv::cuda::GpuMat& image_input, cv::cuda::GpuMat& image_output, trans& transformations, int& width_screen, int& height_screen){
	#if(_HOM_DEBUG) || (_HOM_TIMEIT)
	std::cerr << "### hom <start> ###" << std::endl;
	#endif
// Faster backward homography, mapping by masking and matrix indices method # 0.007 seconds
	#if(_HOM_TIMEIT)
	gputimer watch;
	watch.start();
	#endif

	static const int height_in	= image_input.size().height;
	static const int width_in	= image_input.size().width;
	static const int channels	= image_input.channels();
	static const float xc_in	= (width_in-1)/2;
	static const float yc_in	= (height_in-1)/2;
	
	Eigen::Matrix3f H, Hi;
	H	= calchomography(width_in,height_in,transformations);
	int trans_x, trans_y, trans_z;
	trans_x = transformations.tx;
	trans_y = transformations.ty;
	trans_z = transformations.tz;
	//H	<<	1,0,0,
	//		0,1,0,
	//		0,0,1;
//	Determine dimensions of output image
	Eigen::Vector4f rectangle = box_out(H, width_in, height_in);
	Hi	= H.inverse(); // correct
	#if(_HOM_DEBUG)
	std::cerr <<"Rectangle:\n"<< rectangle << "\n"; // rectangle is xmin, ymin, width, height
	std::cerr << "H:\n" << H << std::endl;
	std::cerr << "Hi:\n" << Hi << std::endl;
	#endif
	float xmin, ymin, xmax, ymax, xc_map, yc_map; // xc and yc are the offsets of the center of the mapping from the upper left corner
	int width_out, height_out;
	xmin		= rectangle[0];
	ymin		= rectangle[1];
	width_out	= int(ceil(rectangle[2])); // in pixels
	height_out	= int(ceil(rectangle[3])); // in pixels
	xmax		= xmin + rectangle[2] - 1; // zero based, thus -1, use rectangle, because it is still float
	ymax		= ymin + rectangle[3] - 1;// zero based, thus -1, use rectangle because float.
	xc_map		= fabs(xmin); // x location of the center in the mapped image
	yc_map		= fabs(ymin); // y location of the center in the mapped image
	//std::cerr << "xc_map: " << xc_map << std::endl;
	//std::cerr << "yc_map: " << yc_map << std::endl;
// Determine size of matrices for performing the mapping.
// Mapping must stay within max size.
// Max size of mapping matrices is minimum of outgoing image dimensions and the max dimensions
	int width_max	= std::min(width_out, width_screen);
	int height_max	= std::min(height_out, height_screen);
	#if(_HOM_DEBUG)
	std::cerr << "xmin:       "	<< xmin			<< std::endl;
	std::cerr << "xmax:       "	<< xmax 		<< std::endl;
	std::cerr << "ymin:       "	<< ymin 		<< std::endl;
	std::cerr << "ymax:       "	<< ymax 		<< std::endl;
	std::cerr << "width_out:  "	<< width_out	<< std::endl;
	std::cerr << "height_out: "	<< height_out	<< std::endl;
	std::cerr << "width_max:  "	<< width_max	<< std::endl;
	std::cerr << "height_max: "	<< height_max	<< std::endl;
	#endif
	#if(_HOM_TIMEIT)
	watch.lap("Mapping Preliminaries");
	#endif
	map(Hi, image_input, image_output, width_out, height_out, xc_in, yc_in, xc_map, yc_map, trans_x, trans_y); // image in and image out are pointers
	
////###################
	#if(_HOM_TIMEIT)
	watch.lap("Perform Mapping");
	#endif
	#if (_HOM_DEBUG) || (_HOM_TIMEIT)
	watch.stop();
	std::cerr << "### hom <end> ###" << std::endl;
	#endif
}
#endif
