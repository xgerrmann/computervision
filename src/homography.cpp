#include "homography.hpp"
// script to test and perform homographies on an example image
// script based on: http://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog/1886060#1886060

rotations calcrotationmatrix(double rx, double ry, double rz){
//	source: http://nghiaho.com/?page_id=846
//	source: https://en.wikipedia.org/wiki/3D_projection (uses negative angles?)
	rx = -rx;
	ry = -ry;
	rz = -rz;

	Eigen::Matrix3f Rx;
	Rx <<	1.0,		0.0,		0.0,
			0.0,		cos(rx),	-sin(rx),
			0.0,		sin(rx),	cos(rx);
	Eigen::Matrix3f	Ry;
	Ry <<	cos(ry),	0,			sin(ry),
			0,			1,			0,
			-sin(ry),	0,			cos(ry);
	Eigen::Matrix3f Rz;
	Rz <<	cos(rz),	-sin(rz),	0,
			sin(rz),	cos(rz),	0,
			0,			0,			1;
	rotations rot;
	rot.Rx=Rx;
	rot.Ry=Ry;
	rot.Rz=Rz;
	return rot;
}

int calchomography(cv::Mat image, double rx, double ry, double rz){
	double pitch = 0.2625*(std::pow(10.0,-3.0)); // [m] pixel pitch (pixel size) assume square pixels, which is generally true
	printf("Pitch: %.6f\n",pitch);

	int height	= image.size().height;
	int width	= image.size().width;
	int channels= image.channels();
	rotations rot = calcrotationmatrix(rx,ry,rz);
	//Rt = Rz*Ry*Rx
	Eigen::Matrix3f Rt	= rot.Rz*rot.Ry*rot.Rx;
	Eigen::Matrix3f Rti	= Rt.inverse();
//	# define 3 points on the virtual image plane
	Eigen::Vector3f p0, p1, p2, pr0, pr1, pr2;
	p0 << 0, 0, 0;
	p1 << 1, 0, 0;
	p2 << 0, 1, 0;
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
	cp0	<< 0,		0;
	cp1	<< width,	0;
	cp2	<< width,	height;
	cp3	<< 0,		height;
	std::vector<Eigen::Vector2f> corners_p;
	corners_p.push_back(cp0);
	corners_p.push_back(cp1);
	corners_p.push_back(cp2);
	corners_p.push_back(cp3);
//	#print corners_p
//	# meters
	double	f, wx, hy;
	f = 0.3;	// [m] Distance of viewer to the screen
	Eigen::Vector3f	e;
	e << 0,0,f;			// position of viewer (user)
//	#print e
	wx	= width*pitch;	// [m]
	hy	= height*pitch;	// [m]
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
//		find the projection of each corner point on the plane
//		note: origin is still center of the plane
		Eigen::Matrix3f A, Ai;
		Eigen::Matrix<float,3,2> tmp_mat;
		Eigen::Vector3f comb, intersection, coords;
		Eigen::Vector2f tmp_vec, corner;
		float x, y;

		std::cerr << "Corner: " << cornerlines.at(i) << "\n";
		A	<< cornerlines.at(i) , -pr1, -pr2;
		std::cerr << "A: \n" << A <<"\n";
		Ai		= A.inverse();
		std::cerr << "Ai: \n" << Ai <<"\n";
		std::cerr << "e\n" << e <<"\n";
		comb	= Ai*(-e);
		std::cerr << "Comb:\n" << comb << "\n";
		tmp_mat << pr1, pr2;
		tmp_vec = comb.block<2,1>(1,0);
		std::cerr << "Comb part:\n" << tmp_vec << "\n";
		intersection = tmp_mat*tmp_vec;
		std::cerr << "Intersection:\n" << intersection << "\n";
//		Compute x,y coordinates in plane by performing the inverse plane rotation on the point of intersection
		coords = Rti*intersection;
		std::cerr << "Coordinates:\n" << coords << "\n";
		// Convert real coordinates into pixed coordinates
		x = (coords[0]+wx/2)/pitch;
		y = -(coords[1]-hy/2)/pitch; //# change y direction
		std::cerr << "x: " << x << "\n";
		std::cerr << "y: " << y << "\n";
		corner << x,y;
		corners_proj.push_back(corner);
	}
//
//	# projected corners is in pixels
//	#print 'Projected corners:\n',corners_proj
//	
//	xmin_out	= np.inf
//	ymin_out	= np.inf
//	xmax_out	= -np.inf
//	ymax_out	= -np.inf
//	for corner_proj in corners_proj:
//		x = corner_proj[0]
//		y = corner_proj[1]
//		if x < xmin_out:
//			xmin_out = x
//		if x > xmax_out:
//			xmax_out = x
//		if y < ymin_out:
//			ymin_out = y
//		if y > ymax_out:
//			ymax_out = y
//	xmin_out = int(np.ceil(xmin_out))
//	ymin_out = int(np.ceil(ymin_out))
//	xmax_out = int(np.ceil(xmax_out))
//	ymax_out = int(np.ceil(ymax_out))
//	height_out	= int(np.ceil(ymax_out - ymin_out))
//	width_out	= int(np.ceil(xmax_out - xmin_out))
//
//	# corners in the projection are now known.
//	# calculate the homography
//	# source: http://math.stackexchange.com/questions/494238/how-to-compute-homography-matrix-h-from-corresponding-points-2d-2d-planar-homog
//	M1 = np.zeros((0,8))
//	M2 = np.zeros((8,1))
//	for i in range(4):
//		xA	= float(corners_p[i][0])
//		yA	= float(corners_p[i][1])
//		xB	= float(corners_proj[i][0])
//		yB	= float(corners_proj[i][1])
//		#print 'xA: ',xA,', xB: ',xB
//		#print 'yA: ',yA,', yB: ',yB
//		row0	= np.matrix([xA,yA,1,0,0,0,-xA*xB,-yA*xB])
//		row1	= np.matrix([0,0,0,xA,yA,1,-xA*yB,-yA*yB])
//		M1 = np.vstack((M1,row0))
//		M1 = np.vstack((M1,row1))
//		M2[i*2,0]	= xB
//		M2[i*2+1,0]	= yB
//	#print M1
//	#print M2
//	H = np.linalg.inv(np.transpose(M1)*M1)*(np.transpose(M1)*M2)
//	#print H
//	H = np.vstack((H,[1]))
//	H = np.reshape(H,(3,3))
//	Hi = np.linalg.inv(H)
//	return (H,Hi,height_out,width_out,xmin_out,xmax_out,ymin_out,ymax_out)
	return 0;
}
//
//def hom0(image,rx,ry,rz):
//## CV2 proprietary method # 0.003 seconds
//	(H,Hi,height_out,width_out,xmin_out,xmax_out, ymin_out,ymax_out) = calchomography(image,rx,ry,rz)
//	t = time.time()
//	(height, width, channels) = image.shape
//	image_out	= cv2.warpPerspective(image,H,(width,height))
//	print 'Hom0: Elapsed time',time.time()-t
//	return image_out
//
//def hom1(image,rx,ry,rz):
//# Own Loop backward homography # 2.06 seconds
//	t = time.time()
//	(H,Hi,height_out,width_out,xmin_out,xmax_out, ymin_out,ymax_out) = calchomography(image,rx,ry,rz)
//	(height, width, channels) = image.shape
//	image_out = np.zeros((height_out,width_out,channels),dtype=np.uint8)
//	for h in range(ymin_out,ymax_out):
//		for w in range(xmin_out,xmax_out):
//			tmp = np.matrix([[w],[h],[1]])
//			res = Hi*tmp
//			scale	= res[2]
//			xtmp	= int(res[0]/scale)
//			ytmp	= int(res[1]/scale)
//			if xtmp<0 or xtmp>=width or ytmp<0 or ytmp>=height:
//				continue
//			else:
//				#print 'y: %+5d -> %+5d'%(ytmp,h)
//				#print 'x: %+5d -> %+5d'%(xtmp,w)
//				#print image[ytmp,xtmp,:]
//				image_out[h-ymin_out,w-xmin_out,:] = image[ytmp,xtmp,:]
//				#print image_out[h,w,:]
//				#print image[ytmp,xtmp,:]
//	print 'Hom1: Elapsed time',time.time()-t
//	return image_out
//
//def hom2(image,rx,ry,rz):
//## Faster backward homography # 0.106 seconds
//	t = time.time()
//	(H,Hi,height_out,width_out,xmin_out,xmax_out, ymin_out,ymax_out) = calchomography(image,rx,ry,rz)
//	(height, width, channels) = image.shape
//	# calc mapping
//	x = range(xmin_out,xmax_out)
//	y = range(ymin_out,ymax_out)
//	X, Y = np.meshgrid(x,y)
//	W = np.ones((height_out,width_out))
//	O = np.stack((X,Y,W),2)
//	map_out	= np.einsum('kp,ijp->ijk',Hi,O)
//	# perform mapping
//	image_out = np.zeros((height_out,width_out,channels),dtype=np.uint8)
//	for h in range(height_out):
//		for w in range(width_out):
//				scale = map_out[h,w,2]
//				x = int(round(map_out[h,w,0]/scale))
//				y = int(round(map_out[h,w,1]/scale))
//				if x<0 or x>=width or y<0 or y>=height:
//					continue
//				image_out[h,w,:] = image[y,x,:]
//	print 'Hom2: Elapsed time',time.time()-t
//	return image_out
//
cv::Mat hom3(cv::Mat image, double rx, double ry, double rz){
// Faster backward homography, mapping by masking and matrix indices method # 0.007 seconds
//	t = time.time()
	//(H,Hi,height_out,width_out,xmin_out,xmax_out, ymin_out,ymax_out) = calchomography(image,rx,ry,rz)
	int H = calchomography(image,rx,ry,rz);
	cv::Mat image_out;
//	(height, width, channels) = image.shape
//	# calc mapping
//	x = range(xmin_out,xmax_out)
//	y = range(ymin_out,ymax_out)
//	X, Y = np.meshgrid(x,y)
//	W = np.ones((height_out,width_out))
//	O = np.stack((X,Y,W),2)
//	map_out	= np.einsum('kp,ijp->ijk',Hi,O)
//	
//	map_out[:,:,1]	= map_out[:,:,1]/map_out[:,:,2]
//	map_out[:,:,0]	= map_out[:,:,0]/map_out[:,:,2]
//	map_out			= map_out.astype(int) # mapping must be of integer type because it is used directly for indexing
//	# construct empty image
//	image_out = np.zeros((height_out,width_out,channels),dtype=np.uint8)
//	# make conditional mask for the width (not larger than max image width)
//	mask_width = np.logical_and(map_out[:,:,0] < width,map_out[:,:,0]>=0)
//	# make conditional mask for the width (not larger than max image height)
//	mask_height = np.logical_and(map_out[:,:,1]>=0,map_out[:,:,1]<height)
//	# combine masks
//	mask_total = np.logical_and(mask_width==1,mask_height==1)
//	# Use mask to copy values from original image
//	image_out[mask_total,:] = image[map_out[mask_total,1],map_out[mask_total,0],:]
//	print 'Hom3: Elapsed time',time.time()-t
	return image_out;
}

int main(){
	cv::Mat image;
	image = cv::imread(default_image);
	std::string windowname	= "Original image";
	cv::namedWindow(windowname);
	cv::imshow(windowname,image);
	cv::waitKey(100);
	
	double	rx = 0.0*PI;
	double	ry = 0.0*PI;
	double	rz = 0.0*PI;
	
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

//#	Hom0: Elapsed time 0.00331997871399
//#	Hom1: Elapsed time 2.2098929882
//#	Hom2: Elapsed time 0.0985150337219
//#	Hom3: Elapsed time 0.00763010978699
	return 0;
}
