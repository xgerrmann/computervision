//cudafuncs.cu

#include "cudafuncs.hpp"

static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

__global__ void calcmap_cuda(int *xp_c, int *yp_c, int *wp_c, float *mxp_c, float *myp_c, float *h_c, int width, int height){
	//int cuda_index = blockDim.x*blockIdx.x + threadIdx.x;
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	// Check if within image bounds
	if((c>=(width))||(r>=(height))) return;
	int cuda_index = r*(width)+c;
	// First calculate the scale, for the X and Y must be devicd by the scale.
	float w				= (h_c[2]*xp_c[cuda_index]+h_c[5]*yp_c[cuda_index]+h_c[8]*wp_c[cuda_index]);
	// x/w
	mxp_c[cuda_index]	= (h_c[0]*xp_c[cuda_index]+h_c[3]*yp_c[cuda_index]+h_c[6]*wp_c[cuda_index])/w;
	// y/w
	myp_c[cuda_index]	= (h_c[1]*xp_c[cuda_index]+h_c[4]*yp_c[cuda_index]+h_c[7]*wp_c[cuda_index])/w;
}

__global__ void domap_cuda(unsigned char *image_out, unsigned char *image_in, float *xp_c, float *yp_c, int *width, int *height){
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	// Check if within image bounds
	//if((c>=(*width))||(r>=(*height))) return;
	int cuda_index = r*(*width)+c;
	//image_out[cuda_index] = image_in[cuda_index];
	image_out[cuda_index] = 90;
}

__global__ void copy_cuda(unsigned char *input,
							unsigned char *output,
							int width,
							int height,
							int step_in,
							int step_out)
{
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	if((xIndex<width) && (yIndex<height)){
		//const int index = yIndex*step + (3*xIndex);
		const int index_in			= yIndex*step_in	+ (3*xIndex);
		const int index_out			= yIndex*step_out	+ (3*xIndex);
		//const int index_out			= yIndex*outputStep	+ (3*xIndex);
		//const int index_out	= yIndex*step_out	+ (3*xIndex);
		//const int index		= xIndex*step + yIndex;
		// make indexes correct
		output[index_out]	= input[index_in];
		output[index_out+1]	= input[index_in+1];
		output[index_out+2]	= input[index_in+2];
	}
	// no return
}

// Partial wrapper for the __global__ calls
void calcmapping(Eigen::MatrixXf& Mx, Eigen::MatrixXf& My,  Eigen::Matrix3f& Hi, int xmin_out, int ymin_out, int wmax, int hmax){
	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### calcmapping <start> ###" << std::endl;
	#endif
	// Get the properties of the GPU device, this will only be executed once.
	static cudaDeviceProp cuda_properties;
	static cudaError_t cuda_error= cudaGetDeviceProperties(&cuda_properties,0); // cuda properties of device 0
	static int N_BLOCKS_MAX		= cuda_properties.maxThreadsPerBlock;	// x dimension
	static int N_THREADS_MAX	= cuda_properties.maxGridSize[0];		// x dimension
	static int N_PIXELS_MAX		= N_BLOCKS_MAX * N_THREADS_MAX;
	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "N_BLOCKS_MAX: " << N_BLOCKS_MAX << std::endl;
	std::cerr << "N_THREADS_MAX:" << N_THREADS_MAX << std::endl;
	#endif
	//#if(_CUDAFUNCS_TIMEIT)
	//gputimer watch;
	//watch.start();
	//#endif

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
	hp = Hi.data(); // Hi is a pointer to an eigen matrix
	
	// Number of rows and columns in Mx and My must be identical
	// TODO: Actually this does not have to be the case!!
	assert(Mx.rows() == My.rows() && Mx.cols() == My.cols());
	// Get pointers to data of mapping matrices
	mxp = Mx.data();	// Mx is a pointer, thus child accessing with ->
	myp = My.data();	// My is a pointer, thus child accessing with ->
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Cuda prelims: ");
	//#endif
	// Allocate space on device for device copies
	cudaMalloc((void **)&xp_c,size_i);
	cudaMalloc((void **)&yp_c,size_i);
	cudaMalloc((void **)&wp_c,size_i);
	cudaMalloc((void **)&mxp_c,size_i);
	cudaMalloc((void **)&myp_c,size_i);
	cudaMalloc((void **)&h_c,size_h);
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Allocate space on device: ");
	//#endif
	// Copy inputs to device
	SAFE_CALL(cudaMemcpy(xp_c,	xp,	size_i,	cudaMemcpyHostToDevice),"CUDA Copy Host To Device Fail");
	SAFE_CALL(cudaMemcpy(yp_c,	yp,	size_i,	cudaMemcpyHostToDevice),"CUDA Copy Host To Device Fail");
	SAFE_CALL(cudaMemcpy(wp_c,	wp,	size_i,	cudaMemcpyHostToDevice),"CUDA Copy Host To Device Fail");
	SAFE_CALL(cudaMemcpy(h_c,	hp,	size_h,	cudaMemcpyHostToDevice),"CUDA Copy Host To Device Fail");
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Copy mem host -> device: ");
	//#endif
	// Execute combine on cpu
	//std::cerr << "Execute device code." << std::endl;
	//calcmap_cuda<<<n_blocks,n_threads>>>(xp_c, yp_c, wp_c, mxp_c, myp_c, h_c);
	// Launch 2D grid
	// Source: http://www.informit.com/articles/article.aspx?p=2455391
//	int TX = 32;
//	int TY = 32;
//	dim3 blockSize(TX, TY);
//	//int bx = (wmax+ blockSize.x-1)/blockSize.x;
//	//int by = (hmax+ blockSize.y-1)/blockSize.y;
//	int bx = (wmax+ TX - 1)/TX;
//	int by = (wmax+ TY - 1)/TY; // Correct? or hmax??
//	dim3 gridSize = dim3 (bx, by);
//	
//	calcmap_cuda<<<gridSize, blockSize>>>(xp_c, yp_c, wp_c, mxp_c, myp_c, h_c, &wmax, &hmax);
	
	// Specify block size
	const dim3 block(16,16);
	// Calculate grid size to cover whole image
	const dim3 grid((wmax + block.x-1)/block.x, (hmax + block.y-1)/block.y);
	calcmap_cuda<<<grid, block>>>(xp_c, yp_c, wp_c, mxp_c, myp_c, h_c, wmax, hmax);
	// Synchronize to check for kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Execute device code: ");
	//#endif
	// copy results to host
	//std::cerr << "Copy memory from device to host." << std::endl;
	SAFE_CALL(cudaMemcpy(mxp, mxp_c, size_f, cudaMemcpyDeviceToHost),"CUDA Copy Device To Host Fail");
	SAFE_CALL(cudaMemcpy(myp, myp_c, size_f, cudaMemcpyDeviceToHost),"CUDA Copy Device To Host Fail");
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Copy mem device -> host: ");
	//#endif
	// cleanup device memory
	//cudaFree(mxp_c);	cudaFree(myp_c),	cudaFree(h_c);
	//cudaFree(xp_c);		cudaFree(yp_c);		cudaFree(wp_c);
	SAFE_CALL(cudaFree(mxp_c) ,"CUDA Free Failed");
	SAFE_CALL(cudaFree(myp_c) ,"CUDA Free Failed");
	SAFE_CALL(cudaFree(xp_c) ,"CUDA Free Failed");
	SAFE_CALL(cudaFree(yp_c) ,"CUDA Free Failed");
	SAFE_CALL(cudaFree(wp_c) ,"CUDA Free Failed");
	//cudaDeviceReset();

	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### calcmapping <end> ###" << std::endl;
	#endif
	// Return nothing, void function.
}


// ######################################################################################
void copy(const cv::Mat& image_in, cv::Mat& image_out){
// This function uploads the input image onto the device (GPU) and downloads it to the
// output image. // TODO: do 2d upload and download for less data transfer.
	//int device = 0;
	//SAFE_CALL(cudaSetDevice(device),"CUDA Set Device Failed");
	//SAFE_CALL(cudaFree(0),"CUDA Free Failed");
	//SAFE_CALL(cudaDeviceSynchronize(),"CUDA Device Sync Failed");
	//SAFE_CALL(cudaThreadSynchronize(),"CUDA Thread Sync Failed");

	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### domapping <start> ###" << std::endl;
	#endif
	cv::imshow("image_input",	image_in);
	cv::imshow("image_out",		image_out);
	cv::waitKey(0);
	// calculate nubmer of bytes in input and output image
	const int inputBytes	= image_in.step*image_in.rows;
	const int outputBytes	= image_out.step*image_out.rows;
	unsigned char *d_input, *d_output;
	//std::cerr	<< "Rows input:           " << image_in.rows			<< std::endl;
	//std::cerr	<< "Cols input:           " << image_in.cols 			<< std::endl;
	//std::cerr	<< "Type input:           " << image_in.type()			<< std::endl;
	//std::cerr	<< "input continuous:     " << image_in.isContinuous()	<< std::endl;
	//std::cerr	<< "Step input:           " << image_in.step			<< std::endl;
	//std::cerr	<< "Rows image_out:       " << image_out.rows			<< std::endl;
	//std::cerr	<< "Cols image_out:       " << image_out.cols			<< std::endl;
	//std::cerr	<< "Type image_out:       " << image_out.type()			<< std::endl;
	//std::cerr	<< "image_out continuous: " << image_out.isContinuous()	<< std::endl;
	//std::cerr	<< "Step image_out:       " << image_out.step			<< std::endl;
	// Allocate device memory
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input,	inputBytes),	"CUDA Malloc input Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output,	outputBytes) ,	"CUDA Malloc output Failed");

	// Copy image_in to device
	SAFE_CALL(cudaMemcpy(d_input, image_in.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output, image_out.ptr(), outputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify block size
	const dim3 block(16,16);
	// Calculate grid size to cover whole image
	const dim3 grid((image_in.cols + block.x-1)/block.x, (image_in.rows + block.y-1)/block.y);
	
	// Launch kernel
	copy_cuda<<<grid,block>>>(d_input,
								d_output,
								image_in.cols,
								image_in.rows,
								image_in.step,
								image_out.step);
	// Synchronize to check for kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
	
	// Retrieve image_input from device
	SAFE_CALL(cudaMemcpy(image_out.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Device To Host Failed");
	
	// Free memory
	SAFE_CALL(cudaFree(d_input) ,"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
	//cudaDeviceReset();
}

// ######################################################################################
void domapping(const cv::Mat& image_input, cv::Mat& image_output, Eigen::MatrixXf& Mx, Eigen::MatrixXf& My){
// domapping
// Function that performs the actual mapping
// d_ stands for device	(gpu)
// h_ stands for host	(cpu)

	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### domapping <start> ###" << std::endl;
	#endif
	//const cv::Mat image_in = cv::imread("media/50x50.png",CV_LOAD_IMAGE_COLOR);
	//copy(image_input,image_output);
	
	// Determine size in bytes of data
	const int inputBytes	= image_input.step*image_input.rows;	// sizeof(uchar) = 1
	const int outputBytes	= image_output.step*image_output.rows;	// sizeof(uchar) = 1
	int N					= Mx.rows()*My.cols();	// number of pixels
	const int mxBytes		= N*sizeof(float);
	const int myBytes		= N*sizeof(float);

	// Create pointers for device data
	unsigned char *d_input, *d_output;
	float *d_mx, *d_my;
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input,	inputBytes),	"CUDA Malloc input Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output,	outputBytes) ,	"CUDA Malloc output Failed");
	SAFE_CALL(cudaMalloc<float>(&d_mx,	mxBytes),	"CUDA Malloc input Failed");
	SAFE_CALL(cudaMalloc<float>(&d_my,	myBytes),	"CUDA Malloc output Failed");

	// Copy to device
	SAFE_CALL(cudaMemcpy(d_input, image_input.ptr(), inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output, image_output.ptr(), outputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_mx, Mx.data(), mxBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_my, My.data(), myBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	
	// Specify block size
	const dim3 block(16,16);
	// Calculate grid size to cover whole image
	// Operate only on region of interest
	const int width		= Mx.cols();
	const int height	= Mx.rows();
	const dim3 grid((width + block.x-1)/block.x, (height + block.y-1)/block.y);
	
	// Launch kernel
	copy_cuda<<<grid,block>>>(d_input,
								d_output,
								width,
								height,
								image_input.step,
								image_output.step);
	
	// Synchronize to check for kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
	
	// Retrieve image_input from device
	SAFE_CALL(cudaMemcpy(image_output.ptr(), d_output, outputBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Device To Host Failed");
	
	// Free memory
	SAFE_CALL(cudaFree(d_input) ,"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output),"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_mx) ,"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_my) ,"CUDA Free Failed");
	
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.stop();
	//#endif
	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### domapping <end> ###" << std::endl;
	#endif
	// Return nothing, void function.
	return;
}
