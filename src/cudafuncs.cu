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
	//if((c>=(*width))||(r>=(*height))) return;
	int cuda_index = r*(*width)+c;
	//image_out[cuda_index] = image_in[cuda_index];
	image_out[cuda_index] = 90;
}

// Partial wrapper for the __global__ calls
extern "C" void calcmapping(Eigen::MatrixXf *Mx, Eigen::MatrixXf *My,  Eigen::Matrix3f *Hi, int xmin_out, int ymin_out, int wmax, int hmax){
	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### calcmapping <start> ###" << std::endl;
	#endif
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
	
	#if(_CUDAFUNCS_DEBUG)
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
	// TODO: Actually this does not have to be the case!!
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

	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### calcmapping <end> ###" << std::endl;
	#endif
	// Return nothing, void function.
	return;
}

extern "C" void domapping(cv::Mat& image_output, const cv::Mat& image_input, Eigen::MatrixXf *Mx, Eigen::MatrixXf *My){
	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### domapping <start> ###" << std::endl;
	#endif
	// d_ stands for device	(gpu)
	// h_ stands for host	(cpu)
	// Copy image_input to device
	uchar *d_input;
	const int input_bytes = image_input.step*image_input.rows;
	cudaMalloc<unsigned char>(&d_input, input_bytes);
	cudaMemcpy(d_input, image_input.ptr(), input_bytes, cudaMemcpyHostToDevice);

	// Retrieve image_input from device
	cv::Mat input_out(image_input.rows, image_input.cols, CV_8UC3);
	cudaMemcpy(input_out.ptr(), d_input, input_bytes, cudaMemcpyDeviceToHost);

	std::cerr << "Show Image." << std::endl;
	cv::imshow("h_input",input_out);
	cv::waitKey(0);
	std::cerr << "Image shown." << std::endl;
//	#if(_CUDAFUNCS_TIMEIT)
//	gputimer watch;
//	watch.start();
//	#endif
//	//cv::imshow("im_in",*image_in);
//	//cv::imshow("im_out",*image_out);
//	//cv::waitKey(0);
//	// TODO: upload input and output matrix to GPU
//	// TODO: inputmage dimensions and Mx and My not correpsonding when rotations and translations are zero.	 
//	int width_mx	= Mx->cols();
//	int height_mx	= Mx->rows();
//	int width_my	= My->cols();
//	int height_my	= My->rows();
//	int width_in	= image_in->cols;
//	int height_in	= image_in->rows;
//	int width_out	= image_out->cols;
//	int height_out	= image_out->rows;
//	int N_m			= width_mx*height_mx;
//	int N_in		= width_in*height_in;
//	int N_out		= width_out*height_out;
//	int channels	= image_in->channels();
//	// Determine size of memory for each input and output
//	int size_m	= N_m*sizeof(float);			// size of Mx and My (one channel)
//	int size_in	= N_in*sizeof(uchar)*channels;	// size of image_in	 (three channels)
//	int size_out= N_out*sizeof(uchar)*channels;	// size of image_out (three channels)
//	#if(_CUDAFUNCS_DEBUG)
//	std::cerr << "Width_mx:      " << width_mx		<< std::endl;
//	std::cerr << "Height_mx:     " << height_mx		<< std::endl;
//	std::cerr << "Width_my:      " << width_my		<< std::endl;
//	std::cerr << "Height_my:     " << height_my		<< std::endl;
//	std::cerr << "Width_in:      " << width_in		<< std::endl;
//	std::cerr << "Height_in:     " << height_in		<< std::endl;
//	std::cerr << "Width_out:     " << width_out		<< std::endl;
//	std::cerr << "Height_out:    " << height_out	<< std::endl;
//	std::cerr << "Channels:      " << channels		<< std::endl;
//	std::cerr << "size_m:        " << size_m		<< std::endl;
//	std::cerr << "size_in:       " << size_in		<< std::endl;
//	std::cerr << "size_out:      " << size_out		<< std::endl;
//	std::cerr << "sizeof(uchar): " << sizeof(uchar)	<< std::endl;
//	std::cerr << "sizeof(cv::CV_8U): " << sizeof(CV_8U)	<< std::endl;
//	std::cerr << "sizeof(float): " << sizeof(float)	<< std::endl;
//	std::cerr << "type image_in: " << image_in->type() << std::endl;
//	std::cerr << "type image_out:" << image_out->type() << std::endl;
//	#endif

	// TODO, keep Mx and My on CUDA device?
	// Create pointers
	//float *mxp, *myp, *mxp_c, *myp_c;
	//uchar *im_out_c, *im_in_c, *im_in, *im_out;
	// Get pointers to data of mapping matrices
	//mxp		= Mx->data();		// Mx is a pointer, thus child accessing with ->
	//myp		= My->data();		// My is a pointer, thus child accessing with ->
	////im_in	= image_in->data;	// Get pointer from cv::Mat
	//im_out	= image_out->data;	// Get pointer fomr cv::Mat
	////im_in	= image_in->ptr(0);	// Get pointer from cv::Mat
	////im_out	= image_out->ptr(0);	// Get pointer fomr cv::Mat
	//std::cerr << "sizeof(im_in): "	<< sizeof(im_in[0])	<< std::endl;
	//std::cerr << "sizeof(im_out): "	<< sizeof(im_out[0])	<< std::endl;
	
	// Allocate space on device for device copies
	//cudaMalloc((void **)&mxp_c,		size_m);
	//cudaMalloc((void **)&myp_c,		size_m);
	//cudaMalloc((void **)&im_in_c,	size_in);
	//cudaMalloc((void **)&im_out_c,	size_out);
	// Copy inputs to device
	//cudaMemcpy(mxp_c,	mxp,	size_m,		cudaMemcpyHostToDevice);
	//cudaMemcpy(myp_c,	myp,	size_m,		cudaMemcpyHostToDevice);
	//cudaMemcpy(im_in_c,	im_in,	size_in,	cudaMemcpyHostToDevice);
//#	std::cerr << "Make GpuMat." << std::endl;
//#	cv::cuda::GpuMat image_in_c;
//#	std::cerr << "Upload Image." << std::endl;
//#	image_in_c.upload(*image_in);
	//uchar *image_in_c;
	//cudaMalloc((void **)&im_in_c,	size_in);

	//std::cerr << "Data host -> device." << std::endl;
	//cudaMemcpy2D(image_in_c.data, image_in_c.step, image_in->data, image_in->step, image_in->cols*image_in->elemSize(), image_in->rows,	cudaMemcpyHostToDevice);

//	// Launch 2D grid
//	// Source: http://www.informit.com/articles/article.aspx?p=2455391
//	int TX = 32;
//	int TY = 32;
//	dim3 blockSize(TX, TY);
//	//int bx = (wmax+ blockSize.x-1)/blockSize.x;
//	//int by = (hmax+ blockSize.y-1)/blockSize.y;
//	int bx = (width_out+ TX - 1)/TX*channels;
////	int by = (width_out+ TY - 1)/TY*channels;
//	int by = (height_out+ TY - 1)/TY*channels;
//	std::cerr << "bx: " << bx << ", by: " << by << std::endl;
//	dim3 gridSize = dim3 (bx, by);
//	//domap_cuda<<<gridSize, blockSize>>>(im_out_c, im_in_c, mxp_c, myp_c, &width_out, &height_out);
//	#if(_CUDAFUNCS_TIMEIT)
//	watch.lap("Execute mapping on device: ");
//	#endif
//
//	// TODO compare pointers to data
//	//std::cerr << "Pointer to host data:   " << im_in << std::endl;
//	//std::cerr << "Pointer to device data: " << im_in_c << std::endl;
//
//	std::cerr << "type image_in: " << image_in->type() << std::endl;
//	std::cerr << "Image in (zeros): "<<std::endl;
//	image_in->setTo(0);
//	for(int i = N_in*10; i < N_in; i++){
//		std::cerr << int(im_in[i]) << std::endl;
//	}
//	std::cerr << "type image_in: " << image_in->type() << std::endl;
//	// Get results back from host
//	//cudaMemcpy(im_out,	im_out_c,	size_out,	cudaMemcpyDeviceToHost);
//	//cudaMemcpy(im_in,	im_in_c,	size_in,	cudaMemcpyDeviceToHost);
//	std::cerr << "Step: " << step << std::endl;
//	//cudaMemcpy2D(im_in, 0, im_in_c, step, sizeof(uchar)*width_in, sizeof(uchar)*height_in,	cudaMemcpyDeviceToHost);
//#	std::cerr << "data device -> host." << std::endl;
//#	cudaMemcpy2D(image_in->data, image_in->step, image_in_c.data, image_in_c.step, image_in_c.cols*image_in_c.elemSize(), image_in_c.rows,	cudaMemcpyDeviceToHost);
//#
//#
//#	// TODO: this does not work.
//#	std::cerr << "Show images." << std::endl;
//#	cv::imshow("im_in",*image_in);	// correct
//#//	cv::imshow("im_out",*image_out);// incorrect
//#	cv::waitKey(0);
//#	// cleanup device memory
//#	//cudaFree(mxp_c);	cudaFree(myp_c);	cudaFree(im_in_c); cudaFree(im_out_c);
//#	//cudaFree(im_in_c);
//#
//#	#if(_CUDAFUNCS_TIMEIT)
//#	watch.stop();
//#	#endif
//#	#if(_CUDAFUNCS_DEBUG)
//#	std::cerr << "### domapping <end> ###" << std::endl;
//#	#endif
	// Return nothing, void function.
	return;
}
