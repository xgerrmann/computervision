//cudafuncs.cu

#include "cudafuncs.hpp"

__global__ void calcmap_cuda(int *xp_c, int *yp_c, int *wp_c, float *mxp_c, float *myp_c, float *mwp_c, float *h_c){
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
	mxp_c[blockIdx.x] = float(xp_c[blockIdx.x]);
	myp_c[blockIdx.x] = float(yp_c[blockIdx.x]);
	mwp_c[blockIdx.x] = float(wp_c[blockIdx.x]);
}

// Partial wrapper for the __global__ calls
extern "C" std::vector<Eigen::MatrixXf> calcmapping(Eigen::Matrix3f Hi, int xmin_out, int ymin_out, int wmax, int hmax){
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
	float	*mxp, *myp, *mwp, *h, *mxp_c, *myp_c, *mwp_c, *h_c;
	
	// Allocate space (and set pointers) for host copies
	xp = X.memptr(); // pointer to x matrix input data
	std::cerr << xp[0] << std::endl;
	std::cerr << xp[1] << std::endl;
	std::cerr << xp[N-2] << std::endl;
	std::cerr << xp[N-1] << std::endl;
	std::cerr << xp[N] << std::endl; // out of bounds, should be random = correct
	yp = Y.memptr(); // pointer to y matrix input data
	std::cerr << yp[0] << std::endl;
	std::cerr << yp[1] << std::endl;
	std::cerr << yp[N-2] << std::endl;
	std::cerr << yp[N-1] << std::endl;
	std::cerr << yp[N] << std::endl; // out of bounds, should be random = correct
	wp = W.memptr(); // pointer to w matrix input data
	h  = Hi.data();	 // Hi is an eigen matrix
	
	Eigen::MatrixXf Mx(hmax, wmax);// = Eigen::Matrix<float,hmax,wmax>::Zero();
	Eigen::MatrixXf My(hmax, wmax);// = Eigen::Matrix<float,hmax,wmax>::Zero();
	Eigen::MatrixXf Mw(hmax, wmax);// = Eigen::Matrix<float,hmax,wmax>::Zero();
//	Eigen::Matrix<float, hmax, wmax> Mx= Eigen::Matrix<float, hmax, wmax>::Zero();// = Eigen::Matrix<float,hmax,wmax>::Zero();
	// Allocate space on host for results
	//mxp = (float *)malloc(size_i); // pointer to x matrix output data
	mxp = Mx.data(); // pointer to x matrix output data
	myp = My.data(); // pointer to x matrix output data
	mwp = Mw.data(); // pointer to x matrix output data

	// Allocate space on device for device copies
	cudaMalloc((void **)&xp_c,size_i);
	cudaMalloc((void **)&yp_c,size_i);
	cudaMalloc((void **)&wp_c,size_i);
	cudaMalloc((void **)&mxp_c,size_i);
	cudaMalloc((void **)&myp_c,size_i);
	cudaMalloc((void **)&mwp_c,size_i);
	cudaMalloc((void **)&h_c,9*sizeof(float));

	// Copy inputs to device
	cudaMemcpy(xp_c,	xp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(yp_c,	yp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(wp_c,	wp,	size_i,	cudaMemcpyHostToDevice);
	cudaMemcpy(h_c,		h,	9*sizeof(float),	cudaMemcpyHostToDevice);

	// Execute combine on cpu
	std::cerr << "Execute device code." << std::endl;
	calcmap_cuda<<<N,1>>>(xp_c, yp_c, wp_c, mxp_c, myp_c, mwp_c, h_c);
	std::cerr << "Finished device code." << std::endl;

	// copy results to host
	std::cerr << "Copy memory from device to host." << std::endl;
	cudaMemcpy(mxp, mxp_c, size_f, cudaMemcpyDeviceToHost);
	cudaMemcpy(myp, myp_c, size_f, cudaMemcpyDeviceToHost);
	cudaMemcpy(mwp, mwp_c, size_f, cudaMemcpyDeviceToHost);

	std::cerr << mxp[0] << std::endl;
	std::cerr << mxp[1] << std::endl;
	std::cerr << mxp[N-2] << std::endl;
	std::cerr << mxp[N-1] << std::endl;

	// Copy to arma matrix
	//arma::Mat<float> Mx(mxp, hmax, wmax, true, true); // Copies memory to matrix
	//arma::Mat<float> My(myp, hmax, wmax, true, true); // Copies memory to matrix
	//arma::Mat<float> Mw(mwp, hmax, wmax, true, true); // Copies memory to matrix
	//arma::Mat<float> Mx(mxp, hmax, wmax); // Copies memory to matrix
	//arma::fcolvec Mx(mxp, N, false, false); // Copies memory to matrix
	//Eigen::Map<Eigen::MatrixXf> Mx(mxp, hmax, wmax);
	//arma::Mat<float> My(myp, hmax, wmax); // Copies memory to matrix
	//arma::Mat<float> Mw(mwp, hmax, wmax); // Copies memory to matrix

	std::cerr << Mx(0) << std::endl;
	std::cerr << Mx(1) << std::endl;
	std::cerr << Mx(N-2) << std::endl;
	std::cerr << Mx(N-1) << std::endl;
	std::cerr << Mx(N) << std::endl;

	std::cerr << "size Mx: " << sizeof(Mx) << std::endl;
	std::cerr << "size Mx[0]: " << sizeof(Mx(0)) << std::endl;
	std::cerr << "size My: " << sizeof(My) << std::endl;
	std::cerr << "size Mw: " << sizeof(Mw) << std::endl;
	
	//std::cerr << "size Mx: " << arma::size(Mx) << std::endl;
	//std::cerr << "size My: " << arma::size(My) << std::endl;
	//std::cerr << "size Mw: " << arma::size(Mw) << std::endl;
	
	std::cerr << "Mx:\n" << Mx << std::endl;
	std::cerr << "My:\n" << My << std::endl;
	std::cerr << "Mw:\n" << Mw << std::endl;
	// TODO M = stack eigen ...
	std::cerr << "Finished printing." << std::endl;	
	//arma::Cube<float> M(join_slices(join_slices(Mx,My),Mw));
	std::vector<Eigen::MatrixXf> M;
	M.push_back(Mx);
	M.push_back(My);
	M.push_back(Mw);

	// cleanup device memory
	cudaFree(mxp_c);	cudaFree(myp_c);	cudaFree(mwp_c);
	cudaFree(xp_c);		cudaFree(yp_c);		cudaFree(wp_c);

	return M;
}

