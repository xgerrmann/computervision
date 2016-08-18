#include <iostream>

//#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "paths.hpp"
#include <math.h>
#include <chrono>
#include <ctime>
// ## Eigen
#include <Eigen/Dense>
#include <limits.h> // for max values of datatypes
// ## Armadillo
#include <armadillo>
//Attention tracker
#include "../include/attention-tracker/src/head_pose_estimation.hpp"

#define EVER ;;


typedef struct {
	Eigen::Matrix3f Rx;
	Eigen::Matrix3f Ry;
	Eigen::Matrix3f Rz;
} rotations;

cv::Mat hom(cv::Mat image, Eigen::Matrix4f pose); // head_pose is a 4x4 cv::Mat doubles
rotations calcrotationmatrix(double rx, double ry, double rz);
Eigen::Matrix3f calchomography(int width, int height, Eigen::Matrix4f pose);
Eigen::Vector4i calccorners(Eigen::Matrix3f H, int height, int width);

const double PI		= 3.141592653589793;
const double INF	= abs(1.0/0.0);
const float PITCH	= 0.2625*(std::pow(10.0,-3.0)); // [m] pixel pitch (pixel size) assume square pixels, which is generally true
// TODO: pitch is different per device

////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// script to test and perform homographies on an example image
// script based on: http://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog/1886060#1886060

rotations calcrotationmatrix(float rx, float ry, float rz){
//	source: http://nghiaho.com/?page_id=846
//	source: https://en.wikipedia.org/wiki/3D_projection (uses negative angles?)
//	Results of rotation matrices are consistent with python script
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
	rotations rot;
	rot.Rx=Rx;
	rot.Ry=Ry;
	rot.Rz=Rz;
	return rot;
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

Eigen::Matrix3f calchomography(int width, int height, Eigen::Matrix4f pose){
// This function calculates the homography matrix, given the rotations rx,ry,rz.
// Coordinate system:
// ^ y+
// |
// --->x

	// Width and height of an image must be > 0
	assert(width>0&&height>0);

	//rotations rot = calcrotationmatrix(rx,ry,rz);
	//Rt = Rz*Ry*Rx
	//Eigen::Matrix3f Rt	= rot.Rz*rot.Ry*rot.Rx;
	std::cerr << "Pose:\n" << pose << std::endl;
	std::cerr << "Rt:\n" << pose.block<3,3>(0,0) << std::endl;
	Eigen::Matrix3f Rt	= pose.block<3,3>(0,0); // extract upper left 3x3 block from the pose, this is the rotation matrix
	std::cerr << "Rt:\n" << Rt << std::endl;
//	std::cerr<<"Rt:\n"<<Rt<<std::endl;
	Eigen::Matrix3f Rti	= Rt.inverse();
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

cv::Mat hom(cv::Mat image, Eigen::Matrix4f pose){
// Faster backward homography, mapping by masking and matrix indices method # 0.007 seconds
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	int height	= image.size().height;
	int width	= image.size().width;
	int channels= image.channels();
	
	Eigen::Matrix3f H, Hi;
	
	H	= calchomography(width,height,pose);
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
	// calc mapping
// meshgrid implementation from:	https://forum.kde.org/viewtopic.php?f=74&t=90876
	arma::Mat<int> x = arma::linspace<arma::Row<int>>(xmin_out,xmax_out,width_out);
	arma::Mat<int> X = arma::repmat(x,height_out,1);
//	x.print("x:") ;
//	X.print("X:") ;
	arma::Mat<int> y = arma::linspace<arma::Col<int>>(ymin_out,ymax_out,height_out);
	arma::Mat<int> Y = arma::repmat(y,1,width_out);
//	y.print("y:") ;
//	Y.print("Y:") ;
	arma::Mat<int> W = arma::ones<arma::Mat<int>>(height_out,width_out);
//	W.print("W:") ;
	arma::Cube<int> O = arma::Cube<int>(height_out, width_out, 2);
	O = arma::join_slices(arma::join_slices(X,Y),W);
	//O.print("O:");// This is always correct.
//	map_out	= np.einsum('kp,ijp->ijk',Hi,O)
//	TODO: Hi naar arma type?
  arma::Cube<float> M = arma::zeros<arma::Cube<float>>(height_out, width_out, 3);
	//M.print("M:");
	for(int i = 0; i < height_out; i++){
		for(int j = 0; j < width_out; j++){
			for(int k = 0; k < 3; k++){
				for(int p = 0; p < 3; p++){
					//TODO: vector operation instead of loop for last loop.
					M(i,j,k) += Hi(k,p)*float(O(i,j,p));
				}
			}
		}
	}
// Following method is slower
//	for(int i = ymin_out; i <= ymax_out; i++){
//		for(int j = xmin_out; j < xmax_out; j++){
//			for(int k = 0; k < 3; k++){
//				//TODO: vector operation instead of loop for last loop.
//				M(i,j,k) = Hi(k,0)*float(j) + Hi(k,1)*float(i) + M(i,j,k) + Hi(k,2); // x,y,w (w=1) 
//				//TODO: or use i and j as values to use and do not construct O
//			}
//		}
//	}
	// Element wise division by scale
	M.slice(0)	= M.slice(0)/M.slice(2);
	M.slice(1)	= M.slice(1)/M.slice(2);
	//M.print("M:");
	// Round is very important in this conversion, otherwise major errors
	// TODO: Check if this still works for negative values (if necessary)
	arma::Cube<int> Mi = arma::conv_to<arma::Cube<int>>::from(round(M)); // mapping must be of integer type because it is used directly for indexing
	//Mi.print("Mi:");
	//M.slice.print("M:");
//	# construct empty image
	cv::Mat image_out	= cv::Mat::zeros(height_out, width_out,CV_8UC3);
	image_out			= cv::Scalar(0,0,0); //fill with zeros, TODO: can be done in previous step. 
	
	int xtmp,ytmp,trans_x, trans_y;
	trans_x = int(round(width/2));
	trans_y = int(round(height/2));
	for(int h = 0; h<height_out; h++){
		for(int w = 0; w<width_out; w++){
			// Change origin from center of image to upper right corner.
			xtmp = Mi(h,w,0)+trans_x;
			ytmp = Mi(h,w,1)+trans_y;
			if(xtmp<0 || xtmp >= width || ytmp<0 || ytmp>=height){
				//std::cerr<<"NOT x:"<<xtmp<<"->"<<w<<std::endl;
				//std::cerr<<"NOT y:"<<ytmp<<"->"<<h<<std::endl;
				continue;
			}
			//std::cerr<<"x:"<<xtmp<<"->"<<w<<std::endl;
			//std::cerr<<"y:"<<ytmp<<"->"<<h<<std::endl;
			//std::cerr<<"y:"<<h<<",x:"<<w<<",R:"<<int(image_arma(h,w,0))<<std::endl;
			//std::cerr<< "R:"<<int(image_arma(ytmp,xtmp,0)) << "==" << int(image_out_arma(h,w,0))<< std::endl;
		//	std::cerr<< int(image_arma(ytmp,xtmp,1)) << "==" << int(image_out_arma(h,w,1))<< std::endl;
		//	std::cerr<< int(image_arma(ytmp,xtmp,2)) << "==" << int(image_out_arma(h,w,2))<< std::endl;
			((uchar*)image_out.data)[(w+h*width_out)*3]		= ((uchar)image.data[(ytmp*width+xtmp)*3]) ;
			((uchar*)image_out.data)[(w+h*width_out)*3+1]	= ((uchar)image.data[(ytmp*width+xtmp)*3+1]) ;
			((uchar*)image_out.data)[(w+h*width_out)*3+2]	= ((uchar)image.data[(ytmp*width+xtmp)*3+2]) ;
		//	std::cerr<< int(image_arma(ytmp,xtmp,0)) << "==" << int(image_out_arma(h,w,0))<< std::endl;
		//	std::cerr<< int(image_arma(ytmp,xtmp,1)) << "==" << int(image_out_arma(h,w,1))<< std::endl;
		//	std::cerr<< int(image_arma(ytmp,xtmp,2)) << "==" << int(image_out_arma(h,w,2))<< std::endl;
		//  Values seem to be correct
		}
	}
	
	std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
	std::cerr << std::printf ("Homography: Elapsed time %f",elapsed_seconds.count())<<std::endl;
	//std::cerr << "Finished" << std::endl;
	return image_out;
}
