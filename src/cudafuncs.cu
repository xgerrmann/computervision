//cudafuncs.cu

#include "cudafuncs.hpp"

__global__ void calcmap_cuda(float *xp_c, float *yp_c, float *wp_c, float *mxp_c, float *myp_c, float *mwp_c, float *h_c){
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
//
}

// Partial wrapper for the __global__ calls
extern "C" arma::Cube<float> calcmapping(Eigen::Matrix3f Hi, int xmin_out, int ymin_out, int wmax, int hmax){
	int xmax,ymax;
	xmax = wmax-xmin_out-1;
	ymax = hmax-ymin_out-1;

	// Prepare inputs
	arma::Mat<int> x = arma::linspace<arma::Row<int> >(xmin_out,xmax,wmax);
	arma::Mat<int> X = arma::repmat(x,hmax,1);
	
	arma::Mat<int> y = arma::linspace<arma::Col<int> >(ymin_out,ymax,hmax);
	arma::Mat<int> Y = arma::repmat(y,1,wmax);
	
	arma::Mat<int> W = arma::ones<arma::Mat<int> >(hmax,wmax);

	int N		= hmax*wmax;
	int size	= N*sizeof(float);
	
	int *xp, *yp, *wp;
	float *mxp, *myp, *mwp, *h;
	float *xp_c, *yp_c, *wp_c, *mxp_c, *myp_c, *mwp_c, *h_c;
	
	// Allocate space (and set pointers) for host copies
	xp = X.memptr(); // pointer to x matrix input data
	yp = Y.memptr(); // pointer to y matrix input data
	wp = W.memptr(); // pointer to w matrix input data
	h  = Hi.data();	 // Hi is an eigen matrix
	
	mxp = (float *)malloc(size); // pointer to x matrix output data
	myp = (float *)malloc(size); // pointer to y matrix output data
	mwp = (float *)malloc(size); // pointer to w matrix output data

	// Allocate space on device for device copies
	cudaMalloc((void **)&xp_c,size);
	cudaMalloc((void **)&yp_c,size);
	cudaMalloc((void **)&wp_c,size);
	cudaMalloc((void **)&mxp_c,size);
	cudaMalloc((void **)&myp_c,size);
	cudaMalloc((void **)&mwp_c,size);
	cudaMalloc((void **)&h_c,9*sizeof(float));

	// Copy inputs to device
	cudaMemcpy(xp_c,	xp,	size,	cudaMemcpyHostToDevice);
	cudaMemcpy(yp_c,	yp,	size,	cudaMemcpyHostToDevice);
	cudaMemcpy(wp_c,	wp,	size,	cudaMemcpyHostToDevice);
	cudaMemcpy(h_c,		h,	size,	cudaMemcpyHostToDevice);

	// Execute combine on cpu
	calcmap_cuda<<<N,1>>>(xp_c, yp_c, wp_c, mxp_c, myp_c, mwp_c, h_c);

	// copy results to host
	cudaMemcpy(mxp, mxp_c, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(myp, myp_c, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(mwp, mwp_c, size, cudaMemcpyDeviceToHost);

	// Copy to arma matrix
	arma::Mat<float> Mx(mxp, hmax, wmax, true); // Copies memory to matrix
	arma::Mat<float> My(myp, hmax, wmax, true); // Copies memory to matrix
	arma::Mat<float> Mw(mwp, hmax, wmax, true); // Copies memory to matrix

	arma::Cube<float> M(join_slices(join_slices(Mx,My),Mw));

	// cleanup
	cudaFree(mxp_c);	cudaFree(myp_c);	cudaFree(mwp_c);
	cudaFree(xp_c);		cudaFree(yp_c);		cudaFree(wp_c);
	free(mxp);			free(myp);			free(mwp);

	return M;
}

