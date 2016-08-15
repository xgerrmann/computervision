#include "homography.hpp"
// script to test and perform homographies on an example image
// script based on: http://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog/1886060#1886060

rotations calcrotationmatrix(float rx, float ry, float rz){
//	source: http://nghiaho.com/?page_id=846
//	source: https://en.wikipedia.org/wiki/3D_projection (uses negative angles?)
//	Results of rotation matrices are consistent with python script
	std::cerr<<"rz: "<<rz<<std::endl;
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
	std::cerr<<"Rz:\n"<<Rz<<std::endl;
//	std::cerr << "Rx:\n"<< Rx<<"\n";
//	std::cerr << "Ry:\n"<< Ry<<"\n";
//	std::cerr << "Rz:\n"<< Rz<<"\n";
	rotations rot;
	rot.Rx=Rx;
	rot.Ry=Ry;
	rot.Rz=Rz;
	return rot;
}

Eigen::Vector4i box_out(Eigen::Matrix3f H, int width, int height){
	Eigen::Vector2f c0, c1, c2, c3;
	c0	<< 0,				0;
	c1	<< float(width),	0;
	c2	<< float(width),	float(height);
	c3	<< 0,				float(height);
	std::vector<Eigen::Vector2f> corners;
	corners.push_back(c0);
	corners.push_back(c1);
	corners.push_back(c2);
	corners.push_back(c3);
	
	std::vector<Eigen::Vector2f> corners_proj;
	float x,y,w;
	for(int i = 0; i<4; i++){
		x = corners.at(i)[0];
		y = corners.at(i)[1];
		w = 1;
		Eigen::Vector3f corner_original, corner_new;
		corner_original << x,y,w;
		std::cerr << "Corner_original:\n" << corner_original << "\n";
		corner_new = H*corner_original;
		std::cerr << "Corner_new:\n" << corner_new << "\n";
		//std::cerr << "Corner_new:\n" << corner_new.block<2,1>(0,0) << "\n";
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
	std::cerr << "xmin: \t" << xmin_out << "\n";
	std::cerr << "xmax: \t" << xmax_out << "\n";
	std::cerr << "ymin: \t" << ymin_out << "\n";
	std::cerr << "ymax: \t" << ymax_out << "\n";
	int height_out, width_out;
	height_out	= ymax_out - ymin_out;
	width_out	= xmax_out - xmin_out;
//	std::cerr << "Height: " << height_out << "\n";
//	std::cerr << "Width:  " << width_out << "\n";
//	std::cerr << "xmin:   " << xmin_out << "\n";
//	std::cerr << "ymin:   " << ymin_out << "\n";
	Eigen::Vector4i rectangle;
	rectangle << xmin_out,ymin_out,width_out,height_out;
	return rectangle; // xmin, ymin, width, height
}

Eigen::Matrix3f calchomography(cv::Mat image, float rx, float ry, float rz){
	float pitch = 0.2625*(std::pow(10.0,-3.0)); // [m] pixel pitch (pixel size) assume square pixels, which is generally true
	printf("Pitch: %.6f\n",pitch);

	int height	= image.size().height;
	int width	= image.size().width;
	int channels= image.channels();
	rotations rot = calcrotationmatrix(rx,ry,rz);
	//Rt = Rz*Ry*Rx
	Eigen::Matrix3f Rt	= rot.Rz*rot.Ry*rot.Rx;
	std::cerr<<"Rt:\n"<<Rt<<std::endl;
	Eigen::Matrix3f Rti	= Rt.inverse();
//	# define 3 points on the virtual image plane
	Eigen::Vector3f p0, p1, p2, pr0, pr1, pr2;
	p0 << 0.0, 0.0, 0.0;
	p1 << 1.0, 0.0, 0.0;
	p2 << 0.0, 1.0, 0.0;
	//std::cerr << p0 << "\n";
	//std::cerr << p1 << "\n";
	//std::cerr << p2 << "\n";
//	# preform rotation of points
	pr0 = Rt*p0;
	pr1 = Rt*p1;
	pr2 = Rt*p2;
//	Find 2 vectors that span the plane:
//	pr0 is always <0,0,0>, so the vectors pr1 and pr2 define the plane

//	Construct the vectors for the view-lines from the optical center to the corners of the virtual image:
//	Corner numbering:
//	0-------1
//	|		|
//	3-------2
//	pixels
//	corner [x,y]
	Eigen::Vector2f cp0, cp1, cp2, cp3;
	cp0	<< 0.0,				0.0;
	cp1	<< float(width),	0.0;
	cp2	<< float(width),	float(height);
	cp3	<< float(0),		float(height);
	std::vector<Eigen::Vector2f> corners_p;
	corners_p.push_back(cp0);
	corners_p.push_back(cp1);
	corners_p.push_back(cp2);
	corners_p.push_back(cp3);
//	#print corners_p
//	# meters
	float	f, wx, hy;
	f = 0.3;	// [m] Distance of viewer to the screen
	Eigen::Vector3f	e;
	e << 0.0,0.0,f;			// position of viewer (user)
//	#print e
	wx	= float(width)*pitch;	// [m]
	hy	= float(height)*pitch;	// [m]
	Eigen::Vector3f c0, c1, c2, c3;
	c0	<< -wx/2,+hy/2,0;// location of corner 0
	c1	<< +wx/2,+hy/2,0;// location of corner 1
	c2	<< +wx/2,-hy/2,0;// location of corner 2
	c3	<< -wx/2,-hy/2,0;// location of corner 3
	c0 = c0-e;// vector from eye to corner 0
	c1 = c1-e;// vector from eye to corner 0
	c2 = c2-e;// vector from eye to corner 0
	c3 = c3-e;// vector from eye to corner 0
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
		// Convert real coordinates into pixed coordinates
		x = (coords[0]+wx/2)/pitch;
		y = -(coords[1]-hy/2)/pitch; //# change y direction
		//std::cerr << "x: " << x << "\n";
		//std::cerr << "y: " << y << "\n";
		corner << x,y;
		corners_proj.push_back(corner);
	}

//	corners in the projection are now known.
//	calculate the homography
//	source: http://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
	Eigen::Matrix<float,8,8> M1;
	Eigen::Matrix<float,8,1> M2;
	float xA, xB, yA, yB;
	for(int i = 0; i<4; i++){
		xA	= corners_p.at(i)[0];	// original corner
		yA	= corners_p.at(i)[1];	// original corner
		xB	= corners_proj.at(i)[0];// new corner (projected)
		yB	= corners_proj.at(i)[1];// new corner (projected)
		//std::cerr << "xA: " << xA <<", xB: " << xB << "\n";
		//std::cerr << "yA: " << yA <<", yB: " << yB << "\n";
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
	std::cerr << "Htmp:\n" << Htmp << "\n";
	// TODO: maybe not transpose?
	H << Htmp; // Fill H with Htmp. First down, then right. Therefore needs to be transposed later.
	std::cerr << "H:\n" << H << "\n";
	H.block<1,1>(2,2) << 1.0;
	std::cerr << "H:\n" << H << "\n";
	H.transposeInPlace(); // in place transposition to avoid aliasing
	std::cerr << "H:\n" << H << "\n";
	// H is calculated correct and results correspond with python script
	return H;
}

//int ut_calchomography(cv::Mat image, double rx,double ry,double rz){
//	H	= calchomography(image,rx,ry,rz);
//	width_out	= 3;
//	height_out	= 3;
//	xmin_out	= 0;
//	xmax_out	= 2;
//	ymin_out	= 0;
//	ymax_out	= 2;
//// linspace etc
//}

cv::Mat hom3(cv::Mat image, float rx, float ry, float rz){
// Faster backward homography, mapping by masking and matrix indices method # 0.007 seconds
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();

	int height	= image.size().height;
	int width	= image.size().width;
	int channels= image.channels();
	
	Eigen::Matrix3f H, Hi;
	H	= calchomography(image,rx,ry,rz);
	//H	<<	1,0,0,
	//		0,1,0,
	//		0,0,1;
	std::cerr << "H:\n" << H << std::endl;
	Eigen::Vector4i rectangle = box_out(H, width, height);
	std::cerr <<"Rectangle:\n"<< rectangle << "\n"; // rectangle is xmin, ymin, width, height
	Hi	= H.inverse();
	int xmin_out, ymin_out, xmax_out, ymax_out, width_out, height_out;
	xmin_out	= rectangle[0];
	ymin_out	= rectangle[1];
	width_out	= rectangle[2];
	height_out	= rectangle[3];
	xmax_out	= xmin_out + width_out-1; // zero based, thus -1
	ymax_out	= ymin_out + height_out-1;// zero based, thus -1
	std::cerr << "xmin: " << xmin_out << std::endl;
	std::cerr << "xmax: " << xmax_out << std::endl;
	std::cerr << "ymin: " << ymin_out << std::endl;
	std::cerr << "ymax: " << ymax_out << std::endl;
	// calc mapping
// meshgrid implementation from:	https://forum.kde.org/viewtopic.php?f=74&t=90876
//	arma::mat Ht = arma::eye(3,3);
//	std::cerr << "Ht:\n" << Ht << std::endl;
//	width_out = width;
//	height_out = height;
//	xmin_out = 0;
//	xmax_out = width-1;
//	ymin_out = 0;
//	ymax_out = height-1;
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
//	O.print("O:");
//	map_out	= np.einsum('kp,ijp->ijk',Hi,O)
//	TODO: Hi naar arma type?
	arma::cube M = arma::zeros(height_out, width_out, 3);
	for(int i = 0; i < height_out; i++){
		for(int j = 0; j < width_out; j++){
			for(int k = 0; k < 3; k++){
				for(int p = 0; p < 3; p++){
					//std::cerr << Hi(k,p) << std::endl;
					//std::cerr << O(i,j,p) << std::endl;
					//TODO: vector operation instead of loop for last loop.
					M(i,j,k) += Hi(k,p)*O(i,j,p);
					//M(i,j,k) += Ht(k,p)*O(i,j,p);
				}
			}
		}
	}
	// Element wise division by scale
	M.slice(0)	= M.slice(0)/M.slice(2);
	M.slice(1)	= M.slice(1)/M.slice(2);
	//M.print("M:");
	arma::Cube<int> Mi = arma::conv_to<arma::Cube<int>>::from(M); // mapping must be of integer type because it is used directly for indexing
	//Mi.print("Mi:");
//	# construct empty image
	arma::Cube<uchar> image_out_arma = arma::zeros<arma::Cube<uchar>>(height_out,width_out,channels);
//	// image is Mat opencv_mat;    //opencv's mat, already transposed.
//	// opencv mat to arma cube, copying is true by default
	std::cerr << "cv -> arma" << std::endl;
	//std::cerr << image << std::endl;
	std::cerr << "Image type:" << image.type() << std::endl;
	// cv::mat to arma
	arma::Cube<uchar> image_arma( image.ptr(), image.rows, image.cols , image.channels());
	// arma to cv::mat
	cv::Mat image_test( height_out, width_out, CV_8UC3, image_arma.memptr());
	// show result
	cv::imshow("Back",image_test);
	// So data cast is correct.
	//image_arma.print("image_arma:");
	std::cerr << "cv -> arma finished" << std::endl;
	std::cerr << size(image_arma) << std::endl;
	int xtmp,ytmp;
	std::cerr << width << std::endl;
	std::cerr << height << std::endl;
	for(int h = 0; h<height_out; h++){
		for(int w = 0; w<width_out; w++){
			xtmp = Mi(h,w,0);
			ytmp = Mi(h,w,1);
			if(xtmp<0 || xtmp >= width || ytmp<0 || ytmp>=height){
				continue;
			}
			//std::cerr<<"x:"<<xtmp<<"->"<<w<<std::endl;
			//std::cerr<<"y:"<<ytmp<<"->"<<h<<std::endl;
			image_out_arma(h,w,0) = image_arma(ytmp,xtmp,0);
			image_out_arma(h,w,1) = image_arma(ytmp,xtmp,1);
			image_out_arma(h,w,2) = image_arma(ytmp,xtmp,2);
		}
	}
	cv::Mat image_out( height_out, width_out, CV_8UC3, image_out_arma.memptr());
	//cv::Mat image_out( height_out, width_out, CV_8UC3, CV_RGB(1,1,1));
	//std::cerr << image_out << std::endl;
	//std::cerr << image_out.rows << ", " << image_out.cols << ", " << image_out.channels() << std::endl;
	std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
	std::cerr << std::printf ("Hom3: Elapsed time %f",elapsed_seconds.count())<<std::endl;
	return image_out;
}

int main(){
	cv::Mat image;
	image = cv::imread(default_image);
	std::string windowname	= "Original image";
	cv::namedWindow(windowname);
	cv::imshow(windowname,image);
	cv::waitKey(100);
	
	float	rx = 0.0*PI;
	float	ry = 0.0*PI;
	float	rz = 0.5*PI;
	
	//im0 = hom0(image,rx,ry,rz)
	//im1 = hom1(image,rx,ry,rz)
	//im2 = hom2(image,rx,ry,rz)
	cv::Mat im3 = hom3(image,rx,ry,rz);

//	Show results
//	cv2.imshow('Hom0',im0)
//	cv2.imshow('Hom1',im1)
//	cv2.imshow('Hom2',im2)
	cv::imshow("Hom3",im3);
	cv::waitKey(0);

// Python:
//#	Hom0: Elapsed time 0.00331997871399
//#	Hom1: Elapsed time 2.2098929882
//#	Hom2: Elapsed time 0.0985150337219
//#	Hom3: Elapsed time 0.00763010978699
	return 0;
}
