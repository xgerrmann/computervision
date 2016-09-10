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

// MAP_CUDA ####################################################################################
// MAP_CUDA ####################################################################################
__global__ void map_cuda(const unsigned char  *d_input,
							unsigned char *d_output,
							float *d_hi,
							int width_input,
							int height_input,
							int width_map,
							int height_map,
							int width_output,
							int height_output,
							int xc_in,
							int yc_in,
							int xc_map,
							int yc_map,
							size_t step_input,
							size_t step_output,
							int trans_x,
							int trans_y)
{
// Width and height are the dimensions of the resulting mapped input image and may vary.
// xIndex and yIndex correspond to the X and Y image-coordinates of the original image AFTER
// mapping.
	
	int col_map = blockIdx.x * blockDim.x + threadIdx.x;
	int row_map = blockIdx.y * blockDim.y + threadIdx.y;

	// ## CALCULATE mapping
	// Early return if outside image bounds
	if((col_map>=width_map)||(row_map>=height_map)) return; // TODO: uncomment
	// X and Y pixel coordinates with the center of the image at (x=0, y=0), then move image back
	// so that upper left pixel is (x=0, y=0)
	float w		=      d_hi[2]*(col_map-xc_map)+d_hi[5]*(row_map-yc_map)+d_hi[8]; // original scale = 1, thus h_c[8]*1 is same as h_c[8]
	int col_in	= int((d_hi[0]*(col_map-xc_map)+d_hi[3]*(row_map-yc_map)+d_hi[6])/w+0.5+xc_in); // +0.5 for decent rounding on conversion to int
	int row_in	= int((d_hi[1]*(col_map-xc_map)+d_hi[4]*(row_map-yc_map)+d_hi[7])/w+0.5+yc_in); // +0.5 for decent rounding on conversion to int

	// if mapped outside original image, then do not compute.
	if((col_in<0) || (col_in>=width_input) || (row_in<0) || (row_in>=height_input)){
		return;
	}
	const int row_map_out = row_map-yc_map-trans_y+height_output/2;// trans_y is positive up
	const int col_map_out = col_map-xc_map+trans_x+width_output/2;

	const int iy_map_out	= row_map_out;	// (float) neccessary for indicating which round to use (double or float)
	const int ix_map_out	= 3*(col_map_out);

	// Return if x or y pixel coordinate falls outside the screen. (Happens because of rounding)
	if((col_map_out >= width_output) || (row_map_out>=height_output) || (col_map_out<0) || (row_map_out<0)) return;
	
	// determine indices
	const int index_out	= iy_map_out*step_output + ix_map_out;
	const int index_in	= row_in*step_input + (3*col_in);

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

// COPY ##########################################################################################
// COPY ##########################################################################################
void copy(const cv::Mat& image_in, cv::Mat& image_out){
// This function uploads the input image onto the device (GPU) and downloads it to the
// output image. // TODO: do 2d upload and download for less data transfer.
	#if(_CUDAFUNCS_DEBUG)
	std::cerr << "### copy <start> ###" << std::endl;
	gputimer watch;
	watch.start();
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

// MAP ###########################################################################################
// MAP ###########################################################################################
void map(Eigen::Matrix3f& Hi, const cv::cuda::GpuMat& image_input, cv::cuda::GpuMat& image_output, int width_map, int height_map, float xc_in, float yc_in, float xc_map, float yc_map, int trans_x, int trans_y){
// domapping
// Function that performs the actual mapping
// d_ stands for device	(gpu)
// h_ stands for host	(cpu)
	#if (_CUDAFUNCS_DEBUG) || (_CUDAFUNCS_TIMEIT)
	std::cerr << "### map <start> ###" << std::endl;
	gputimer watch;
	watch.start();
	#endif
	
	// Determine size in bytes of data
	int size_h	= 9*sizeof(float); // H (in fact a 3x3 matrix) contains 9 float scalars.

	// Create pointer for device data
	float	*h_hi, *d_hi;
	h_hi = Hi.data(); // Hi is a pointer to an eigen matrix
	// Allocate space on device
	SAFE_CALL(cudaMalloc((void **)&d_hi,size_h),	"CUDA Malloc homography");
	#if(_CUDAFUNCS_TIMEIT)
	watch.lap("Allocate space on device");
	#endif

	// Copy to device
	SAFE_CALL(cudaMemcpy(d_hi,	h_hi,	size_h,	cudaMemcpyHostToDevice),"CUDA Copy Host To Device Fail");
	#if(_CUDAFUNCS_TIMEIT)
	watch.lap("Copy data to device");
	#endif
	
	// Specify block size
	//const dim3 block(1,1);
	const dim3 block(16,16);
	//const dim3 block(32,32);
	//const dim3 block(64,64); // too large
	// Calculate grid size to cover whole image
	const int width_out		= image_output.size().width;
	const int height_out	= image_output.size().height;
	const dim3 grid((width_map + block.x-1)/block.x, (height_map + block.y-1)/block.y);
	
	#if(_CUDAFUNCS_DEBUG)
		std::cerr << "width:  " << width_out << std::endl;
		std::cerr << "height: " << height_out << std::endl;
	#endif
	int width_in	= image_input.size().width;
	int height_in	= image_input.size().height;
	// TODO: pass as function arguments instead of calculating here ??
	// Launch kernel
	map_cuda<<<grid,block>>>(image_input.ptr<uchar>(),
								image_output.ptr<uchar>(),
								d_hi,
								width_in,
								height_in,
								width_map,
								height_map,
								width_out,
								height_out,
								xc_in,
								yc_in,
								xc_map,
								yc_map,
								image_input.step,
								image_output.step,
								trans_x,
								trans_y);
	
	// Synchronize to check for kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(),"Kernel Launch Failed");
	#if(_CUDAFUNCS_TIMEIT)
	watch.lap("Execute device code");
	#endif

	// Free memory
	SAFE_CALL(cudaFree(d_hi) ,"CUDA Free Failed");

	#if(_CUDAFUNCS_TIMEIT)
	watch.lap("Free memory in device");
	#endif

	#if(_CUDAFUNCS_TIMEIT)
	watch.stop();
	#endif
	#if (_CUDAFUNCS_DEBUG) || (_CUDAFUNCS_TIMEIT)
	std::cerr << "### map <end> ###" << std::endl;
	#endif
	// Return nothing, void function.
}
