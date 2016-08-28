//cudafuncs.cu

#include "cudafuncs.hpp"

__global__ void calcmap_cuda(int *xp_c, int *yp_c, int *wp_c, float *mxp_c, float *myp_c, float *h_c){
//	//TODO: vector operation instead of loop for last loop.
//	//M(i,j,k) += Hi(k,p)*float(O(i,j,p));
//	double tmp = 0;
//	for(int p = 0; p < 3; p++){
//		tmp += Hi(k,p)*float(O(i,j,p));
//	}
//	M[blockIdx.x] = 
//	for(int k = 0; k < 3; k++){
//		arma::Mat<int> tmpCube(O.tube(i,j));
//		arma::Col<float> tmpvec = arma::conv_to<arma::fcolvec>::from(tmpCube);
//		arma::Col<float> res = Hia(k,arma::span::all)*tmpvec;
//		//M(i,j,k) = res.at(0);
//		M[blockIdx.x+k*N] = res.at(0);
//	}
	float w = float(wp_c[blockIdx.x]);
	mxp_c[blockIdx.x] = float(xp_c[blockIdx.x])/w;
	myp_c[blockIdx.x] = float(yp_c[blockIdx.x])/w;
}

// Partial wrapper for the __global__ calls
extern "C" void calcmapping(Eigen::MatrixXf *Mx, Eigen::MatrixXf *My,  Eigen::Matrix3f Hi, int xmin_out, int ymin_out, int wmax, int hmax){
	std::cerr << "Enter calcmapping." << std::endl;
	int xmax,ymax;
	xmax = wmax-xmin_out-1;
	ymax = hmax-ymin_out-1;

	// Prepare inputs
	arma::Mat<int> x = arma::linspace<arma::Row<int> >(xmin_out,xmax,wmax);
	arma::Mat<int> X = arma::repmat(x,hmax,1);
	X.print("X:");	
	arma::Mat<int> y = arma::linspace<arma::Col<int> >(ymin_out,ymax,hmax);
	arma::Mat<int> Y = arma::repmat(y,1,wmax);
	Y.print("Y:");	
	
	arma::Mat<int> W = arma::ones<arma::Mat<int> >(hmax,wmax);
	W.print("W:");	

	int N		= hmax*wmax;
	std::cerr << "hmax: " << hmax << ", wmax: "<< wmax << std::endl;
	std::cerr << "N: " << N << std::endl;
	std::cerr << "size X: " << arma::size(X) << std::endl;
	std::cerr << "size Y: " << arma::size(Y) << std::endl;
	std::cerr << "size W: " << arma::size(W) << std::endl;
	std::cerr << "size X: " << sizeof(X) << std::endl;
	std::cerr << "size Y: " << sizeof(Y) << std::endl;
	std::cerr << "size W: " << sizeof(W) << std::endl;
	int size_i	= N*sizeof(int);
	int size_f	= N*sizeof(float);
	std::cerr << "sizeof(int): " << sizeof(int) << std::endl;
	std::cerr << "sizeof(float): " << sizeof(float) << std::endl;
	std::cerr << "size_i: " << size_i << std::endl;
	std::cerr << "size_f: " << size_f << std::endl;
	
	int		*xp, *yp, *wp, *xp_c, *yp_c, *wp_c;
	float	*mxp, *myp, *h, *mxp_c, *myp_c, *h_c;
	
	// Allocate space (and set pointers) for host copies
	xp = X.memptr(); // pointer to x matrix input data
	yp = Y.memptr(); // pointer to y matrix input data
	wp = W.memptr(); // pointer to w matrix input data
	h  = Hi.data();	 // Hi is an eigen matrix
	
	// Number of rows and columns in Mx and My must be identical
	assert(Mx.rows() == My.rows() && Mx.cols() == My.cols());
	// Get pointers to data of mapping matrices
	mxp = Mx->data();	// Mx is a pointer, thus child accessing with ->
	myp = My->data();	// My is a pointer, thus child accessing with ->

	// Allocate space on device for device copies
	cudaMalloc((void **)&xp_c,size_i);
	cudaMalloc((void **)&yp_c,size_i);
	cudaMalloc((void **)&wp_c,size_i);
	cudaMalloc((void **)&mxp_c,size_i);
	cudaMalloc((void **)&myp_c,size_i);
	cudaMalloc((void **)&h_c,9*sizeof(float));

	// Copy inputs to device
	cudaMemcpy(xp_c,	xp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(yp_c,	yp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(wp_c,	wp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(h_c,		h,	9*sizeof(float),	cudaMemcpyHostToDevice);

	// Execute combine on cpu
	std::cerr << "Execute device code." << std::endl;
	calcmap_cuda<<<N,1>>>(xp_c, yp_c, wp_c, mxp_c, myp_c, h_c);
	std::cerr << "Finished device code." << std::endl;

	// copy results to host
	std::cerr << "Copy memory from device to host." << std::endl;
	cudaMemcpy(mxp, mxp_c, size_f, cudaMemcpyDeviceToHost);
	cudaMemcpy(myp, myp_c, size_f, cudaMemcpyDeviceToHost);

	// cleanup device memory
	cudaFree(mxp_c);	cudaFree(myp_c);
	cudaFree(xp_c);		cudaFree(yp_c);		cudaFree(wp_c);

	// No return, void function.
}

