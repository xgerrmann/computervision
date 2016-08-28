//cudafuncs.cu

#include "cudafuncs.hpp"

__global__ void calcmap_cuda(int *xp_c, int *yp_c, int *wp_c, float *mxp_c, float *myp_c, float *h_c){
	// TODO: max number of blocks

	// First calculate the scale, for the X and Y must be devicd by the scale.
	float w				= (h_c[2]*float(xp_c[blockIdx.x])+h_c[5]*float(yp_c[blockIdx.x])+h_c[8]*float(wp_c[blockIdx.x]));
	// x/w
	mxp_c[blockIdx.x]	= (h_c[0]*float(xp_c[blockIdx.x])+h_c[3]*float(yp_c[blockIdx.x])+h_c[6]*float(wp_c[blockIdx.x]))/w;
	// y/w
	myp_c[blockIdx.x]	= (h_c[1]*float(xp_c[blockIdx.x])+h_c[4]*float(yp_c[blockIdx.x])+h_c[7]*float(wp_c[blockIdx.x]))/w;
}

// Partial wrapper for the __global__ calls
extern "C" void calcmapping(Eigen::MatrixXf *Mx, Eigen::MatrixXf *My,  Eigen::Matrix3f *Hi, int xmin_out, int ymin_out, int wmax, int hmax){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds = 0;
	cudaEventRecord(start,0);

	//std::cerr << "Enter calcmapping." << std::endl;
	// Calculate max x and y of image
	int xmax,ymax;
	xmax = xmin_out + wmax - 1;
	ymax = ymin_out + hmax - 1;

	// Prepare inputs for the device code
	// STATIC because every loop this is the same
	// Input are meshgrid MATLAB-like arrays of the X and Y coordinates of the pixels and the scale (=1)
	static arma::Mat<int> x = arma::linspace<arma::Row<int> >(xmin_out,xmax,wmax);
	static arma::Mat<int> X = arma::repmat(x,hmax,1);
	static arma::Mat<int> y = arma::linspace<arma::Col<int> >(ymin_out,ymax,hmax);
	static arma::Mat<int> Y = arma::repmat(y,1,wmax);
	static arma::Mat<int> W = arma::ones<arma::Mat<int> >(hmax,wmax);
	
	#ifdef _DEBUG
	//X.print("X:");
	//Y.print("Y:");
	//W.print("W:");
	#endif
	
	// Determine data sizes
	int N		= hmax*wmax;
	int size_i	= N*sizeof(int);
	int size_f	= N*sizeof(float);
	int size_h	= 9*sizeof(float); // H (in fact a 3x3 matrix) contains 9 float scalars.

	// Create pointers to host and device data
	int		*xp, *yp, *wp, *xp_c, *yp_c, *wp_c;
	float	*mxp, *myp, *hp, *mxp_c, *myp_c, *h_c;
	
	// Link the pointers to the corresponding data
	xp = X.memptr(); // pointer to x matrix input data
	yp = Y.memptr(); // pointer to y matrix input data
	wp = W.memptr(); // pointer to w matrix input data
	hp = Hi->data(); // Hi is a pointer to an eigen matrix
	
	// Number of rows and columns in Mx and My must be identical
	assert(Mx->rows() == My->rows() && Mx->cols() == My->cols());
	// Get pointers to data of mapping matrices
	mxp = Mx->data();	// Mx is a pointer, thus child accessing with ->
	myp = My->data();	// My is a pointer, thus child accessing with ->
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cerr << "Cuda prelims: " << milliseconds/1000 << std::endl;

	// Allocate space on device for device copies
	cudaEventRecord(start,0);
	cudaMalloc((void **)&xp_c,size_i);
	cudaMalloc((void **)&yp_c,size_i);
	cudaMalloc((void **)&wp_c,size_i);
	cudaMalloc((void **)&mxp_c,size_i);
	cudaMalloc((void **)&myp_c,size_i);
	cudaMalloc((void **)&h_c,size_h);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cerr << "Allocate space on device: " << milliseconds/1000 << std::endl;

	cudaEventRecord(start,0);
	// Copy inputs to device
	cudaMemcpy(xp_c,	xp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(yp_c,	yp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(wp_c,	wp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(h_c,		hp,	size_h,	cudaMemcpyHostToDevice);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cerr << "Copy mem host -> device: " << milliseconds/1000 << std::endl;

	// Execute combine on cpu
	//std::cerr << "Execute device code." << std::endl;
	cudaEventRecord(start,0);
	calcmap_cuda<<<N,1>>>(xp_c, yp_c, wp_c, mxp_c, myp_c, h_c);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cerr << "Execute device code: " << milliseconds/1000 << std::endl;
	//std::cerr << "Finished device code." << std::endl;

	// copy results to host
	//std::cerr << "Copy memory from device to host." << std::endl;
	cudaEventRecord(start,0);
	cudaMemcpy(mxp, mxp_c, size_f, cudaMemcpyDeviceToHost);
	cudaMemcpy(myp, myp_c, size_f, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cerr << "Copy mem device -> host: " << milliseconds/1000 << std::endl;

	// cleanup device memory
	cudaFree(mxp_c);	cudaFree(myp_c);
	cudaFree(xp_c);		cudaFree(yp_c);		cudaFree(wp_c);

	// No return, void function.
}

