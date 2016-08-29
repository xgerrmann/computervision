//cudafuncs.cu

#include "cudafuncs.hpp"

__global__ void calcmap_cuda(int *xp_c, int *yp_c, int *wp_c, float *mxp_c, float *myp_c, float *h_c, int *width, int *height){
	// TODO: max number of blocks
	//int cuda_index = blockDim.x*blockIdx.x + threadIdx.x;
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	// Check if within image bounds
	if((c>=(*width))||(r>=(*height))) return;
	int cuda_index = r*(*width)+c;
	// First calculate the scale, for the X and Y must be devicd by the scale.
	float w				= (h_c[2]*xp_c[cuda_index]+h_c[5]*yp_c[cuda_index]+h_c[8]*wp_c[cuda_index]);
	// x/w
	mxp_c[cuda_index]	= (h_c[0]*xp_c[cuda_index]+h_c[3]*yp_c[cuda_index]+h_c[6]*wp_c[cuda_index])/w;
	// y/w
	myp_c[cuda_index]	= (h_c[1]*xp_c[cuda_index]+h_c[4]*yp_c[cuda_index]+h_c[7]*wp_c[cuda_index])/w;
}

__global__ void domap_cuda(uchar *image_out, uchar *image_in, float *xp_c, float *yp_c, int *width, int *height){
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	// Check if within image bounds
	if((c>=(*width))||(r>=(*height))) return;
	int cuda_index = r*(*width)+c;
	image_out[cuda_index] = image_in[cuda_index];
}

// Partial wrapper for the __global__ calls
extern "C" void calcmapping(Eigen::MatrixXf *Mx, Eigen::MatrixXf *My,  Eigen::Matrix3f *Hi, int xmin_out, int ymin_out, int wmax, int hmax){
	// Get the properties of the GPU device, this will only be executed once.
	static cudaDeviceProp cuda_properties;
	static cudaError_t cuda_error= cudaGetDeviceProperties(&cuda_properties,0); // cuda properties of device 0
	static int N_BLOCKS_MAX		= cuda_properties.maxThreadsPerBlock;	// x dimension
	static int N_THREADS_MAX	= cuda_properties.maxGridSize[0];		// x dimension
	static int N_PIXELS_MAX = N_BLOCKS_MAX * N_THREADS_MAX;
	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "N_BLOCKS_MAX: " << N_BLOCKS_MAX << std::endl;
	std::cerr << "N_THREADS_MAX:" << N_THREADS_MAX << std::endl;
	#endif
	#if(_CUDAFUNCS_TIMEIT)
	gputimer watch;
	watch.start();
	#endif

	//std::cerr << "Enter calcmapping." << std::endl;
	// Calculate max x and y of image
	int xmax,ymax;
	xmax = xmin_out + wmax - 1;
	ymax = ymin_out + hmax - 1;

	// Prepare inputs for the device code
	// STATIC because every loop this is the same
	// Input are meshgrid MATLAB-like arrays of the X and Y coordinates of the pixels and the scale (=1)
	arma::Mat<int> x = arma::linspace<arma::Row<int> >(xmin_out,xmax,wmax);
	arma::Mat<int> X = arma::repmat(x,hmax,1);
	arma::Mat<int> y = arma::linspace<arma::Col<int> >(ymin_out,ymax,hmax);
	arma::Mat<int> Y = arma::repmat(y,1,wmax);
	arma::Mat<int> W = arma::ones<arma::Mat<int> >(hmax,wmax);
	
	#if(_CUDAFUNCS_TIMEIT)
	//X.print("X:");
	//Y.print("Y:");
	//W.print("W:");
	#endif
	
	// Determine data sizes
	int N		= hmax*wmax;
	//std::cerr << hmax << "," << wmax << std::endl;
	assert(N<N_PIXELS_MAX);// number of pixels must be smaller then the total number of threads (in the x dimension)
	int size_i	= N*sizeof(int);
	int size_f	= N*sizeof(float);
	int size_h	= 9*sizeof(float); // H (in fact a 3x3 matrix) contains 9 float scalars.

	// determine number of blocks and threads per block
	//int n_blocks	= ceil(float(N)/float(N_THREADS_MAX));
	//int n_threads	= ceil(float(N)/float(n_blocks));
	//int n_threads	= N_THREADS_MAX;
	//std::cerr << "n_blocks:  "<< n_blocks << std::endl;
	//std::cerr << "n_threads: "<< n_threads << std::endl;

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
	#if(_CUDAFUNCS_TIMEIT)
	watch.lap("Cuda prelims: ");
	#endif
	// Allocate space on device for device copies
	cudaMalloc((void **)&xp_c,size_i);
	cudaMalloc((void **)&yp_c,size_i);
	cudaMalloc((void **)&wp_c,size_i);
	cudaMalloc((void **)&mxp_c,size_i);
	cudaMalloc((void **)&myp_c,size_i);
	cudaMalloc((void **)&h_c,size_h);
	#if(_CUDAFUNCS_TIMEIT)
	watch.lap("Allocate space on device: ");
	#endif
	// Copy inputs to device
	cudaMemcpy(xp_c,	xp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(yp_c,	yp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(wp_c,	wp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(h_c,		hp,	size_h,	cudaMemcpyHostToDevice);
	#if(_CUDAFUNCS_TIMEIT)
	watch.lap("Copy mem host -> device: ");
	#endif
	// Execute combine on cpu
	//std::cerr << "Execute device code." << std::endl;
	//calcmap_cuda<<<n_blocks,n_threads>>>(xp_c, yp_c, wp_c, mxp_c, myp_c, h_c);
	// Launch 2D grid
	// Source: http://www.informit.com/articles/article.aspx?p=2455391
	int TX = 32;
	int TY = 32;
	dim3 blockSize(TX, TY);
	//int bx = (wmax+ blockSize.x-1)/blockSize.x;
	//int by = (hmax+ blockSize.y-1)/blockSize.y;
	int bx = (wmax+ TX - 1)/TX;
	int by = (wmax+ TY - 1)/TY; // Correct? or hmax??
	dim3 gridSize = dim3 (bx, by);
	calcmap_cuda<<<gridSize, blockSize>>>(xp_c, yp_c, wp_c, mxp_c, myp_c, h_c, &wmax, &hmax);
	#if(_CUDAFUNCS_TIMEIT)
	watch.lap("Execute device code: ");
	#endif
	// copy results to host
	//std::cerr << "Copy memory from device to host." << std::endl;
	cudaMemcpy(mxp, mxp_c, size_f, cudaMemcpyDeviceToHost);
	cudaMemcpy(myp, myp_c, size_f, cudaMemcpyDeviceToHost);
	#if(_CUDAFUNCS_TIMEIT)
	watch.lap("Copy mem device -> host: ");
	#endif
	// cleanup device memory
	cudaFree(mxp_c);	cudaFree(myp_c);
	cudaFree(xp_c);		cudaFree(yp_c);		cudaFree(wp_c);

	// Return nothing, void function.
	return;
}

extern "C" void domapping(cv::cuda::GpuMat *image_out, cv::cuda::GpuMat *image_in, Eigen::MatrixXf *Mx, Eigen::MatrixXf *My){
	#if(_CUDAFUNCS_TIMEIT)
	gputimer watch;
	watch.start();
	#endif
	int width	= Mx->cols();
	int height	= Mx->rows();
	int N		= width*height;

	//std::cerr << hmax << "," << wmax << std::endl;
	int size_i	= N*sizeof(int);
	// TODO, keep Mx and My on CUDA device?
	// Create pointers
	float *mxp, *myp, *mxp_c, *myp_c;
	uchar *im_out_c, *im_in_c;
	// Get pointers to data of mapping matrices
	mxp		= Mx->data();	// Mx is a pointer, thus child accessing with ->
	myp		= My->data();	// My is a pointer, thus child accessing with ->
	im_in_c	= image_in->ptr<uchar>();	// input image already resides on the GPU, Get device pointer from GPU mat
	im_out_c= image_out->ptr<uchar>();	// output image already resides on the GPU, Get device pointer from GPU mat

	// Allocate space on device for device copies
	cudaMalloc((void **)&mxp_c,size_i);
	cudaMalloc((void **)&myp_c,size_i);
	
	// Copy inputs to device
	cudaMemcpy(mxp_c,	mxp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(myp_c,	myp,	size_i,	cudaMemcpyHostToDevice);

	// Launch 2D grid
	// Source: http://www.informit.com/articles/article.aspx?p=2455391
	int TX = 32;
	int TY = 32;
	dim3 blockSize(TX, TY);
	//int bx = (wmax+ blockSize.x-1)/blockSize.x;
	//int by = (hmax+ blockSize.y-1)/blockSize.y;
	int bx = (width+ TX - 1)/TX;
	int by = (width+ TY - 1)/TY;
	dim3 gridSize = dim3 (bx, by);
	domap_cuda<<<gridSize, blockSize>>>(im_out_c, im_in_c, mxp_c, myp_c, &width, &height);

	#if(_CUDAFUNCS_TIMEIT)
	watch.stop();
	#endif
	// Return nothing, void function.
	return;
}
