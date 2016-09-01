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
__global__ void calcmap_cuda(int *d_mx,
								int *d_my,
								float *hi_c,
								int width_map_out,
								int height_map_out,
								float xc,
								float yc)
{
	const int col = blockIdx.x*blockDim.x + threadIdx.x;
	const int row = blockIdx.y*blockDim.y + threadIdx.y;
	// Early return if outside image bounds
	if((col>=width_map_out)||(row>=height_map_out)) return;
	const int index_col_major = row+col*height_map_out;
	// TODO: convert to int nicely
	// X and Y pixel coordinates with the center of the image at (x=0, y=0), then move image back
	// so that upper left pixel is (x=0, y=0)
	float w					=  hi_c[2]*(col-xc)+hi_c[5]*(row-yc)+hi_c[8]; // original scale = 1, thus h_c[8]*1 is same as h_c[8]
	d_mx[index_col_major]	= int((hi_c[0]*(col-xc)+hi_c[3]*(row-yc)+hi_c[6])/w+xc+0.5); // +0.5 for decent rounding on conversion to int
	d_my[index_col_major]	= int((hi_c[1]*(col-xc)+hi_c[4]*(row-yc)+hi_c[7])/w+yc+0.5); // +0.5 for decent rounding on conversion to int
}

// DOMAP_CUDA ####################################################################################
// DOMAP_CUDA ####################################################################################
__global__ void domap_cuda(unsigned char  *d_input,
							unsigned char *d_output,
							int *d_mx,
							int *d_my,
							int width_input,
							int height_input,
							int width_output,
							int height_output,
							int xc,
							int yc,
							int step_input,
							int step_output)
{
// Width and height are the dimensions of the resulting mapped input image and may vary.
// xIndex and yIndex correspond to the X and Y image-coordinates of the original image AFTER
// mapping.
	
	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	// Determine the index of the mapping matrices
	//const int map_index = yIndex*width+xIndex;
	// TODO: heights and widths
	const int map_index = yIndex + xIndex*(height_input); // column major

	int x_tmp	= d_mx[map_index];
	int y_tmp	= d_my[map_index];

	// if mapped outside original image, then do not compute.
	if((x_tmp<0) || (x_tmp>=width_input) || (y_tmp<0) || (y_tmp>=height_input)){
		return;
	}
	const int row_map_out	= round((float)yIndex-yc+height_output/2);	// (float) neccessary for indicating which round to use (double or float)
	const int col_map_out	= round((float)(xIndex-xc+width_output/2));
	const int iy_map_out	= row_map_out;	// (float) neccessary for indicating which round to use (double or float)
	const int ix_map_out	= 3*col_map_out;

	// Return if x or y pixel coordinate falls outside the screen. (Happens because of rounding)
	if((col_map_out >= width_output) || (row_map_out>=height_output) || (col_map_out<0) || (row_map_out<0)) return;
	
	// determine indices
	const int index_out		= iy_map_out*step_output + ix_map_out;
	const int index_in		= y_tmp*step_input + (3*x_tmp);

	// Perform mapping
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
void calcmapping(Eigen::MatrixXi& Mx, Eigen::MatrixXi& My,  Eigen::Matrix3f& Hi, float x_map_min, float y_map_min){
	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### calcmapping <start> ###" << std::endl;
	#endif
	// Determine size of the resulting mapping
	int height_map_out, width_map_out;
	height_map_out	= Mx.rows();
	width_map_out	= Mx.cols();
	#if(_CUDAFUNCS_DEBUG)
	// Get the properties of the GPU device, this will only be executed once.
	static cudaDeviceProp cuda_properties;
	static cudaError_t cuda_error= cudaGetDeviceProperties(&cuda_properties,0); // cuda properties of device 0
	static int N_BLOCKS_MAX		= cuda_properties.maxThreadsPerBlock;	// x dimension
	static int N_THREADS_MAX	= cuda_properties.maxGridSize[0];		// x dimension
	//static int N_PIXELS_MAX		= N_BLOCKS_MAX * N_THREADS_MAX;
	std::cerr << "N_BLOCKS_MAX: " << N_BLOCKS_MAX << std::endl;
	std::cerr << "N_THREADS_MAX:" << N_THREADS_MAX << std::endl;
	#endif
	//#if(_CUDAFUNCS_TIMEIT)
	//gputimer watch;
	//watch.start();
	//#endif

	
	// Determine data sizes
	int N		= height_map_out*width_map_out;
	//std::cerr << hmax << "," << wmax << std::endl;
	assert(N<N_PIXELS_MAX);// number of pixels must be smaller then the total number of threads (in the x dimension)
	int size_i	= N*sizeof(int);
	//int size_f	= N*sizeof(float);
	int size_h	= 9*sizeof(float); // H (in fact a 3x3 matrix) contains 9 float scalars.

	// Create pointers to host and device data
	float	*h_hi, *d_hi;
	int		*d_mx, *d_my, *h_mx, *h_my;
	// Link the pointers to the corresponding data
	h_hi = Hi.data(); // Hi is a pointer to an eigen matrix
	
	// Get pointers to data of mapping matrices
	h_mx = Mx.data();	// Mx is a pointer, thus child accessing with ->
	h_my = My.data();	// My is a pointer, thus child accessing with ->
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Cuda prelims: ");
	//#endif
	
	// Allocate space on device for device copies
	cudaMalloc((void **)&d_mx,size_i);
	cudaMalloc((void **)&d_my,size_i);
	cudaMalloc((void **)&d_hi,size_h);
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Allocate space on device: ");
	//#endif
	
	// Copy inputs to device
	SAFE_CALL(cudaMemcpy(d_hi,	h_hi,	size_h,	cudaMemcpyHostToDevice),"CUDA Copy Host To Device Fail");
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Copy mem host -> device: ");
	//#endif
	
	// Specify block size
	const dim3 block(16,16);
	// Calculate grid size to cover whole image
	const dim3 grid((width_map_out + block.x-1)/block.x, (height_map_out + block.y-1)/block.y);
	calcmap_cuda<<<grid, block>>>(d_mx, d_my, d_hi, width_map_out, height_map_out, x_map_min, y_map_min);
	// Synchronize to check for kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Execute device code: ");
	//#endif
	
	// Copy results to host
	SAFE_CALL(cudaMemcpy(h_mx, d_mx, size_i, cudaMemcpyDeviceToHost),"CUDA Copy Device To Host Fail");
	SAFE_CALL(cudaMemcpy(h_my, d_my, size_i, cudaMemcpyDeviceToHost),"CUDA Copy Device To Host Fail");
	//#if(_CUDAFUNCS_TIMEIT)
	//watch.lap("Copy mem device -> host: ");
	//#endif
	
	// Cleanup device memory
	SAFE_CALL(cudaFree(d_mx) ,"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_my) ,"CUDA Free Failed");
	SAFE_CALL(cudaFree(d_hi) ,"CUDA Free Failed");
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
void domapping(const cv::Mat& image_input, cv::Mat& image_output, Eigen::MatrixXi& Mx, Eigen::MatrixXi& My, float xc, float yc){
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
	const int mxBytes		= N*sizeof(int);
	const int myBytes		= N*sizeof(int);

	std::cerr << "Mx rows: " << Mx.rows() << ", Mx cols: " << Mx.cols() << std::endl;
	// Create pointers for device data
	unsigned char *d_input, *d_output;
	int *d_mx, *d_my;
	SAFE_CALL(cudaMalloc<unsigned char>(&d_input,	inputBytes),	"CUDA Malloc input Failed");
	SAFE_CALL(cudaMalloc<unsigned char>(&d_output,	outputBytes) ,	"CUDA Malloc output Failed");
	SAFE_CALL(cudaMalloc<int>(&d_mx,	mxBytes),	"CUDA Malloc input Failed");
	SAFE_CALL(cudaMalloc<int>(&d_my,	myBytes),	"CUDA Malloc output Failed");

	// Copy to device
	SAFE_CALL(cudaMemcpy(d_input,	image_input.ptr(),	inputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output,	image_output.ptr(),	outputBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_mx, Mx.data(), mxBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_my, My.data(), myBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	
	// Specify block size
	//const dim3 block(16,16);
	const dim3 block(2,2); //TODO: block, 16x16
	// Calculate grid size to cover whole image
	// Operate only on region of interest
	//const int width_out		= Mx.cols();
	//const int height_out	= Mx.rows();
	const int width_out		= image_output.size().width;
	const int height_out	= image_output.size().height;
	const dim3 grid((width_out + block.x-1)/block.x, (height_out + block.y-1)/block.y);
	
	#if(_CUDAFUNCS_DEBUG)
		std::cerr << "width:  " << width_out << std::endl;
		std::cerr << "height: " << height_out << std::endl;
	#endif
	int width_in	= image_input.size().width;
	int height_in	= image_input.size().height;
	// Launch kernel
	domap_cuda<<<grid,block>>>(d_input,
								d_output,
								d_mx,
								d_my,
								width_in,
								height_in,
								width_out,
								height_out,
								xc,
								yc,
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
