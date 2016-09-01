//cudafuncs.cu

#include "cudafuncs.hpp"

// _SAFE_DUA_CALL ################################################################################
// _SAFE_DUA_CALL ################################################################################
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number)
{
	if(err!=cudaSuccess)
	{
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

// CALCMAP_CUDA ##################################################################################
// CALCMAP_CUDA ##################################################################################
__global__ void calcmap_cuda(float *mxp_c, float *myp_c, float *h_c, int width_in, int height_in, int width_out, int height_out){
	int c = blockIdx.x*blockDim.x + threadIdx.x;
	int r = blockIdx.y*blockDim.y + threadIdx.y;
	// Early return if outside image bounds
	if((c>=(width_out))||(r>=(height_out))) return;
	int index_col_major = r+c*height_out;
	// Translation of image center here
	float t_x = (width_in-1)/2;
	float t_y = (height_in-1)/2;
	// First calculate the scale, for the X and Y must be devicd by the scale.
	float w					=  h_c[2]*(c-t_x)+h_c[5]*(r-t_y)+h_c[8]; // original scale = 1, thus h_c[8]*1 is same as h_c[8]
	mxp_c[index_col_major]	= (h_c[0]*(c-t_x)+h_c[3]*(r-t_y)+h_c[6])/w + t_x;//- t_x;
	myp_c[index_col_major]	= (h_c[1]*(c-t_x)+h_c[4]*(r-t_y)+h_c[7])/w + t_y;//- t_y;
}

// DOMAP_CUDA ####################################################################################
// DOMAP_CUDA ####################################################################################
__global__ void domap_cuda(unsigned char  *d_input,
							unsigned char *d_output,
							float *d_mx,
							float *d_my,
							int width,
							int height,
							int step_input,
							int step_output)
{
// Width and height are the dimensions of the resulting mapped input image and may vary.
// xIndex and yIndex correspond to the X and Y image-coordinates of the original image AFTER
// mapping.
	// transpose (+x, +y)
	//int trans_x = int((width-1)/2+0.5); // +0.5, then cast to int. (same as round and then convert to int)
	//int trans_y = int((height-1)/2+0.5); // -1 because image sizes are 1 px too large // TODO: fix
	
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	// Determine the index of the mapping matrices
	//const int map_index = yIndex*width+xIndex;
	const int map_index = xIndex*(height) + yIndex; // column major

	int x_tmp	= d_mx[map_index];
	int y_tmp	= d_my[map_index];

	// if mapped outside original image, then do not compute
	if((x_tmp<0) || (x_tmp>=width) || (y_tmp<0) || (y_tmp>=height)){
		return;
	}
	const int index_out		= yIndex*step_output + (3*xIndex);
	// Perform mapping
	const int index_in		= y_tmp*step_input + (3*x_tmp);

	d_output[index_out]		= d_input[index_in];
	d_output[index_out+1]	= d_input[index_in+1];
	d_output[index_out+2]	= d_input[index_in+2];
}

// COPY_CUDA #####################################################################################
// COPY_CUDA #####################################################################################
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

// CALCMAPPING  ##################################################################################
// CALCMAPPING  ##################################################################################
void calcmapping(Eigen::MatrixXf& Mx, Eigen::MatrixXf& My,  Eigen::Matrix3f& Hi, int width_in, int height_in, float xmin, float ymin){
	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### calcmapping <start> ###" << std::endl;
	#endif
	// Determine size of the resulting mapping
	int height, width;
	height	= Mx.rows();
	width	= Mx.cols();
	#if(_CUDAFUNCS_DEBUG)
	// Get the properties of the GPU device, this will only be executed once.
	static cudaDeviceProp cuda_properties;
	static cudaError_t cuda_error= cudaGetDeviceProperties(&cuda_properties,0); // cuda properties of device 0
	static int N_BLOCKS_MAX		= cuda_properties.maxThreadsPerBlock;	// x dimension
	static int N_THREADS_MAX	= cuda_properties.maxGridSize[0];		// x dimension
	static int N_PIXELS_MAX		= N_BLOCKS_MAX * N_THREADS_MAX;
	std::cerr << "N_BLOCKS_MAX: " << N_BLOCKS_MAX << std::endl;
	std::cerr << "N_THREADS_MAX:" << N_THREADS_MAX << std::endl;
	#endif
	//#if(_CUDAFUNCS_TIMEIT)
	//gputimer watch;
	//watch.start();
	//#endif

	
	// Determine data sizes
	int N		= height*width;
	//std::cerr << hmax << "," << wmax << std::endl;
	assert(N<N_PIXELS_MAX);// number of pixels must be smaller then the total number of threads (in the x dimension)
	int size_i	= N*sizeof(int);
	int size_f	= N*sizeof(float);
	int size_h	= 9*sizeof(float); // H (in fact a 3x3 matrix) contains 9 float scalars.

	// Create pointers to host and device data
	//int		*xp, *yp, *xp_c, *yp_c;
	float	*mxp, *myp, *hp, *mxp_c, *myp_c, *h_c;
	
	// Link the pointers to the corresponding data
	hp = Hi.data(); // Hi is a pointer to an eigen matrix
	
	// Get pointers to data of mapping matrices
	mxp = Mx.data();	// Mx is a pointer, thus child accessing with ->
	myp = My.data();	// My is a pointer, thus child accessing with ->
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Cuda prelims: ");
	//#endif
	
	// Allocate space on device for device copies
	cudaMalloc((void **)&mxp_c,size_i);
	cudaMalloc((void **)&myp_c,size_i);
	cudaMalloc((void **)&h_c,size_h);
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Allocate space on device: ");
	//#endif
	
	// Copy inputs to device
	SAFE_CALL(cudaMemcpy(h_c,	hp,	size_h,	cudaMemcpyHostToDevice),"CUDA Copy Host To Device Fail");
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Copy mem host -> device: ");
	//#endif
	
	// Specify block size
	const dim3 block(16,16);
	// Calculate grid size to cover whole image
	const dim3 grid((width + block.x-1)/block.x, (height + block.y-1)/block.y);
	calcmap_cuda<<<grid, block>>>(mxp_c, myp_c, h_c, width_in, height_in, width, height);
	// Synchronize to check for kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Execute device code: ");
	//#endif
	
	// Copy results to host
	SAFE_CALL(cudaMemcpy(mxp, mxp_c, size_f, cudaMemcpyDeviceToHost),"CUDA Copy Device To Host Fail");
	SAFE_CALL(cudaMemcpy(myp, myp_c, size_f, cudaMemcpyDeviceToHost),"CUDA Copy Device To Host Fail");
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Copy mem device -> host: ");
	//#endif
	
	// Cleanup device memory
	SAFE_CALL(cudaFree(mxp_c) ,"CUDA Free Failed");
	SAFE_CALL(cudaFree(myp_c) ,"CUDA Free Failed");
	//cudaDeviceReset();

	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### calcmapping <end> ###" << std::endl;
	#endif
	// Return nothing, void function.
}


// COPY ##########################################################################################
// COPY ##########################################################################################
void copy(const cv::Mat& image_in, cv::Mat& image_out){
// This function uploads the input image onto the device (GPU) and downloads it to the
// output image. // TODO: do 2d upload and download for less data transfer.
	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### copy <start> ###" << std::endl;
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
	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### copy <end> ###" << std::endl;
	#endif
}

// DOMAPPING ##################EE#################################################################
// DOMAPPING ##################EE#################################################################
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

	std::cerr << "Mx rows: " << Mx.rows() << ", Mx cols: " << Mx.cols() << std::endl;
	// Create pointers for device data
	unsigned char *d_input, *d_output;
	float *d_mx, *d_my;
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input,	inputBytes),	"CUDA Malloc input Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output,	outputBytes) ,	"CUDA Malloc output Failed");
	SAFE_CALL(cudaMalloc<float>(&d_mx,	mxBytes),	"CUDA Malloc input Failed");
	SAFE_CALL(cudaMalloc<float>(&d_my,	myBytes),	"CUDA Malloc output Failed");

	// Copy to device
	SAFE_CALL(cudaMemcpy(d_input,	image_input.ptr(),	inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output,	image_output.ptr(),	outputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_mx, Mx.data(), mxBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_my, My.data(), myBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	
	// Specify block size
	//const dim3 block(16,16);
	const dim3 block(1,1);
	// Calculate grid size to cover whole image
	// Operate only on region of interest
	const int width_out		= Mx.cols();
	const int height_out	= Mx.rows();
	const dim3 grid((width_out + block.x-1)/block.x, (height_out + block.y-1)/block.y);
	
	#if(_CUDAFUNCS_DEBUG)
		std::cerr << "width:  " << width_out << std::endl;
		std::cerr << "height: " << height_out << std::endl;
	#endif
	// Launch kernel
	domap_cuda<<<grid,block>>>(d_input,
								d_output,
								d_mx,
								d_my,
								width_out,
								height_out,
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
}
