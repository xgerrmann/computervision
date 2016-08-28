// homography.hpp
#ifndef homography_h
#define homography_h

#include <iostream>     // std::cout
//#include <algorithm>    // std::max

#include "../lib/timer/timer.hpp"

//#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "paths.hpp"
#include <math.h>
// ## Eigen
//#include <Eigen/Dense>
#include <eigen3/Eigen/Dense> // << changes
#include <limits.h> // for max values of datatypes
// ## Armadillo
#include <armadillo>
// ## Attention tracker
#include "../include/attention-tracker/src/head_pose_estimation.hpp"
// ## Xlib
#include <X11/Xlib.h> // To determine the display size
// ## cudafuncs
#include "cudafuncs.hpp"


#define EVER ;;

#define _TIMEIT 1

typedef struct {
	Eigen::Matrix3f Rx;
	Eigen::Matrix3f Ry;
	Eigen::Matrix3f Rz;
} Rxyz;

//typedef struct {
//	float dx;
//	float dy;
//	float dz;
//	float rx;
//	float ry;
//	float rz;
//} trans;

typedef std::map<std::string,float> trans;

cv::Mat hom(cv::Mat image, trans transformations, int width_max, int height_max);
Rxyz calcrotationmatrix(double rx, double ry, double rz);
std::vector<float> calcrotations(Eigen::Matrix3f Rt);
Eigen::Matrix3f calchomography(int width, int height, trans transformations);
Eigen::Vector4i calccorners(Eigen::Matrix3f H, int height, int width);
////// Cuda funcitons in cudafuncs.cu
//__global__ void calcmap_cuda(xp_c, yp_c, wp_c, mxp_c, myp_c, mwp_c, h_c);
//arma::Mat calcmapping(float Hi, int xmin_out, int ymin_out, int wmax, int hmax);


const double PI		= 3.141592653589793;
const double INF	= abs(1.0/0.0);
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

std::vector<float> calcrotations(Eigen::Matrix3f Rt){
//	source: http://stackoverflow.com/questions/15022630/how-to-calculate-the-angle-from-roational-matrix
//
//	source: http://www.staff.city.ac.uk/~sbbh653/publications/euler.pdf
//	it is interesting to note that there is always more than one sequence of rotations
//	about the three principle axes that results in the same orientation of an object
	float rx, ry, rz;
	std::vector<float> rots;
	double r11, r21, r31, r32, r33;
	r11 = Rt(0,0);
	r21 = Rt(1,0);
	r31 = Rt(2,0);
	r33 = Rt(2,2);
	rx = (float) atan2( r32, r33);
	ry = (float) atan2(-r31, sqrt(pow(r32,2)+pow(r33,2)));
	rz = (float) atan2( r21, r11);
	rots.push_back(rx);
	rots.push_back(ry);
	rots.push_back(rz);
//	std::cerr << "calcrotations (output): rx, ry, rz: "<< rx << ", "<< ry << ", "<< rz << std::endl;
	return rots;
}

Eigen::Vector4i box_out(Eigen::Matrix3f H, int width_original, int height_original){
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
	int xmin_out, ymin_out, xmax_out, ymax_out;
	float xtmp, ytmp;
	xmin_out	= INT_MAX;
	ymin_out	= INT_MAX;
	xmax_out	= INT_MIN;
	ymax_out	= INT_MIN;
	for(int i = 0; i<4; i++){
		xtmp = corners_proj.at(i)[0];
		ytmp = corners_proj.at(i)[1];
		if(int(round(xtmp)) < xmin_out)
			xmin_out = int(round(xtmp));
		if(int(round(xtmp)) > xmax_out)
			xmax_out = int(round(xtmp));
		if(int(round(ytmp)) < ymin_out)
			ymin_out = int(round(ytmp));
		if(int(round(ytmp)) > ymax_out)
			ymax_out = int(round(ytmp));
	}
//	std::cerr << "xmin: \t" << xmin_out << "\n";
//	std::cerr << "xmax: \t" << xmax_out << "\n";
//	std::cerr << "ymin: \t" << ymin_out << "\n";
//	std::cerr << "ymax: \t" << ymax_out << "\n";
	int height_image_out, width_image_out;
	height_image_out	= ymax_out - ymin_out + 1; // height_image_out is in pixels so +1
	width_image_out		= xmax_out - xmin_out + 1; // width_image_out  is in pixels so +1
//	std::cerr << "Height: " << height_out << "\n";
//	std::cerr << "Width:  " << width_out << "\n";
//	std::cerr << "xmin:   " << xmin_out << "\n";
//	std::cerr << "ymin:   " << ymin_out << "\n";
	Eigen::Vector4i rectangle;
	rectangle << xmin_out,ymin_out,width_image_out,height_image_out;
	return rectangle; // xmin, ymin, width, height
}

Eigen::Matrix3f calchomography(int width, int height, trans transformations){
// This function calculates the homography matrix, given the rotations rx,ry,rz.
// Coordinate system:
// ^ y+
// |
// --->x

	// Width and height of an image must be > 0
	assert(width>0&&height>0);

	//Rt = Rz*Ry*Rx
	Rxyz rot			= calcrotationmatrix(transformations["rx"], transformations["ry"], transformations["rz"]);
	Eigen::Matrix3f Rt	= rot.Rz*rot.Ry*rot.Rx;
	//std::cerr << "Rt:\n"	<< Rt	<< std::endl;
	
	Eigen::Matrix3f Rti	= Rt.inverse();
//	std::cerr << "Rti:\n"	<< Rti	<< std::endl;
//	# define 3 points on the virtual image plane
	Eigen::Vector3f p0, p1, p2, pr0, pr1, pr2;
	p0 << 0.0, 0.0, 0.0;	// [m]
	p1 << 1.0, 0.0, 0.0;	// [m]
	p2 << 0.0, 1.0, 0.0;	// [m]
	//std::cerr << p0 << "\n";
	//std::cerr << p1 << "\n";
	//std::cerr << p2 << "\n";
//	# preform rotation of points
	pr0 = Rt*p0;
	pr1 = Rt*p1;
	pr2 = Rt*p2;
//	Find 2 vectors that span the plane:
//	pr0 is always <0,0,0>, so the vectors pr1 and pr2 define the plane.

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
		float x, y;
		// TODO: more comments
		//std::cerr << "Corner: " << cornerlines.at(i) << "\n";
		A	<< cornerlines.at(i) , -pr1, -pr2;
		//std::cerr << "A: \n" << A <<"\n";
		Ai		= A.inverse();
		//std::cerr << "Ai: \n" << Ai <<"\n";
		//std::cerr << "e\n" << e <<"\n";
		comb	= Ai*(-e);
		//std::cerr << "Comb:\n" << comb << "\n";
		tmp_mat << pr1, pr2;
		tmp_vec = comb.block<2,1>(1,0);
		//std::cerr << "Comb part:\n" << tmp_vec << "\n";
		intersection = tmp_mat*tmp_vec;
		//std::cerr << "Intersection:\n" << intersection << "\n";
		//Compute x,y coordinates in plane by performing the inverse plane rotation on the point of intersection
		coords = Rti*intersection;
		//std::cerr << "Coordinates:\n" << coords << "\n";
		// Convert real coordinates into pixel coordinates
		x = coords[0]/PITCH;	// [pixels]
		y = coords[1]/PITCH;	// [pixels]
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
	return H;
}

cv::Mat hom(cv::Mat image, trans transformations, int width_max, int height_max){
// Faster backward homography, mapping by masking and matrix indices method # 0.007 seconds
	#ifdef _TIMEIT
	timer watch;
	watch.start();
	#endif

	int height	= image.size().height;
	int width	= image.size().width;
	int channels= image.channels();
	
	Eigen::Matrix3f H, Hi;
	//arma::Mat<float> H, Hi;	
	H	= calchomography(width,height,transformations);
	//H	<<	1,0,0,
	//		0,1,0,
	//		0,0,1;
//	std::cerr << "H:\n" << H << std::endl;
//	std::cerr << width << std::endl;
//	std::cerr << height << std::endl;
	Eigen::Vector4i rectangle = box_out(H, width, height);
	//std::cerr <<"Rectangle:\n"<< rectangle << "\n"; // rectangle is xmin, ymin, width, height
	Hi	= H.inverse(); // correct
//	std::cerr << "Hi:\n" << Hi << std::endl;
	int xmin_out, ymin_out, xmax_out, ymax_out, width_out, height_out;
	xmin_out	= rectangle[0];
	ymin_out	= rectangle[1];
	width_out	= rectangle[2];
	height_out	= rectangle[3];
	xmax_out	= xmin_out + width_out-1; // zero based, thus -1
	ymax_out	= ymin_out + height_out-1;// zero based, thus -1
//	std::cerr << "xmin: " << xmin_out << std::endl;
//	std::cerr << "xmax: " << xmax_out << std::endl;
//	std::cerr << "ymin: " << ymin_out << std::endl;
//	std::cerr << "ymax: " << ymax_out << std::endl;
//	std::cerr << "width_out:  " << width_out << std::endl;
//	std::cerr << "height_out: " << height_out << std::endl;

// Determine size of matrices for performing the mapping.
// Mapping must stay within max size.
// Max size of mapping matrices is minimum of outgoing image dimensions and the max dimensions
	int wmax = std::min(width_out, width_max);
	int hmax = std::min(height_out, height_max);
	
	// Mapping is calculated on GPU
	//arma::Cube<float> M = calcmapping(Eigen::Matrix3f Hi, int xmin_out, int ymin_out, int wmax, int hmax);
	Eigen::MatrixXf Mx(hmax, wmax);// = Eigen::Matrix<float,hmax,wmax>::Zero();
	Eigen::MatrixXf My(hmax, wmax);// = Eigen::Matrix<float,hmax,wmax>::Zero();
	calcmapping(&Mx, &My, Hi, xmin_out, ymin_out, wmax, hmax);
	std::cerr << "Mapping calculated." << std::endl;
	std::cerr << "Mx:" << Mx << std::endl;
	// Element wise division by scale TODO
	Mx = Mx.cwiseQuotient(My); // TODO: direct delen door de schaal op GPU
	My = My.cwiseQuotient(My);
	#ifdef _TIMEIT
	watch.lap("Calc Mapping");
	#endif
	//M.print("M:");
	// Round is very important in this conversion, otherwise major errors
	// TODO (solved, answer = yes): Check if this still works for negative values (if necessary)
	//std::vector<Eigen::MatrixXf> Mi = arma::conv_to<arma::Cube<int>>::from(round(M)); // mapping must be of integer type because it is used directly for indexing
	// TODO: to integer via rounding or perform interpolation later.
	//Mi.print("Mi:");
	//M.slice.print("M:");
//	# construct empty image

	cv::Mat image_out	= cv::Mat::zeros(height_max, width_max,CV_8UC3); // 3 channel 8-bit character
	
	int xtmp,ytmp,trans_x, trans_y;
	trans_x = int(round(width/2));
	trans_y = int(round(height/2));
	for(int h = 0; h<hmax; h++){
		for(int w = 0; w<wmax; w++){
	//		std::cerr<<"h:"<<h<<std::endl;
	//		std::cerr<<"height_out:"<<height_out<<std::endl;
	//		std::cerr<<arma::size(Mi);
			// Change origin from center of image to upper right corner.
			xtmp = Mx(h,w)+trans_x;
			ytmp = My(h,w)+trans_y;
			if(xtmp<0 || xtmp >= width || ytmp<0 || ytmp>=height){
				//std::cerr<<"NOT x:"<<xtmp<<"->"<<w<<std::endl;
				//std::cerr<<"NOT y:"<<ytmp<<"->"<<h<<std::endl;
				continue;
			}
		//	std::cerr<<"x:"<<xtmp<<"->"<<w<<std::endl;
		//	std::cerr<<"y:"<<ytmp<<"->"<<h<<std::endl;
			//std::cerr<<"y:"<<h<<",x:"<<w<<",R:"<<int(image_arma(h,w,0))<<std::endl;
			//std::cerr<< "R:"<<int(image_arma(ytmp,xtmp,0)) << "==" << int(image_out_arma(h,w,0))<< std::endl;
		//	std::cerr<< int(image_arma(ytmp,xtmp,1)) << "==" << int(image_out_arma(h,w,1))<< std::endl;
		//	std::cerr<< int(image_arma(ytmp,xtmp,2)) << "==" << int(image_out_arma(h,w,2))<< std::endl;
			((uchar*)image_out.data)[(w+h*width_max)*3]		= ((uchar)image.data[(ytmp*width+xtmp)*3]) ;
			((uchar*)image_out.data)[(w+h*width_max)*3+1]	= ((uchar)image.data[(ytmp*width+xtmp)*3+1]) ;
			((uchar*)image_out.data)[(w+h*width_max)*3+2]	= ((uchar)image.data[(ytmp*width+xtmp)*3+2]) ;
		//	std::cerr << "Test";
		//	std::cerr<< int(image_arma(ytmp,xtmp,0)) << "==" << int(image_out_arma(h,w,0))<< std::endl;
		//	std::cerr<< int(image_arma(ytmp,xtmp,1)) << "==" << int(image_out_arma(h,w,1))<< std::endl;
		//	std::cerr<< int(image_arma(ytmp,xtmp,2)) << "==" << int(image_out_arma(h,w,2))<< std::endl;
		//  Values seem to be correct
		}
	}
	#ifdef _TIMEIT
	watch.lap("Perform Mapping");
	#endif
	//watch.stop("Homography:");
	return image_out;
}
#endif
