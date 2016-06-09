#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <curand.h>
#include <assert.h>
#include <unistd.h>
#include <cublas_v2.h>
#include <iostream>
#include <complex.h>
#include <math.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include "cublas_beamformer.h"

using namespace std;

// Fill the array A(nr_rows_A, nr_cols_A) with random numbers on GPU
void GPU_fill(cuComplex *A, int nr_rows_A, int nr_cols_A) {
	cuComplex *G;
	G = new cuComplex[nr_rows_A*nr_cols_A];
	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
		G[i].x = (i + 1)%(nr_rows_A*nr_cols_A/(N_BIN));
		G[i].y = (i + 1)%(nr_rows_A*nr_cols_A/(N_BIN));

	}

	cudaMemcpy(A,G,nr_rows_A * nr_cols_A * sizeof(cuComplex),cudaMemcpyHostToDevice);
	delete[] G;
}

void GPU_fill2(cuComplex *A, int nr_rows_A, int nr_cols_A) {
	cuComplex *G;
	G = new cuComplex[nr_rows_A*nr_cols_A];
	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
		G[i].x = i%(nr_rows_A*nr_cols_A/(N_BIN));
		G[i].y = i%(nr_rows_A*nr_cols_A/(N_BIN));
	}

	cudaMemcpy(A,G,nr_rows_A * nr_cols_A * sizeof(cuComplex),cudaMemcpyHostToDevice);
	delete[] G;
}

void print_matrix(const cuComplex *A, int nr_rows_A, int nr_cols_A, int nr_sheets_A) {
	for(int i = 0; i < nr_rows_A; ++i){
		for(int j = 0; j < nr_cols_A; ++j){
			for(int k = 0; k < nr_sheets_A; ++k){
				//				cout << A[j * nr_rows_A + i].x << "+" << A[j * nr_rows_A + i].y << "i" <<" ";
				printf("%i,%i,%i: %e + %e i\n",i,j,k,A[k*nr_rows_A*nr_cols_A + j * nr_rows_A + i].x, A[k*nr_rows_A*nr_cols_A + j * nr_rows_A + i].y);
			}
		}
		//			cout << endl;
	}
	//		cout << endl;
	//	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
	//		printf("%i,: %e + %e i\n",i,A[i].x, A[i].y);
	//	}
}


void print_matrix2(const float *A, int nr_rows_A, int nr_cols_A) {
	//	for(int j = 0; j < nr_cols_A; ++j){
	//		for(int i = 0; i < nr_rows_A; ++i){
	//			//cout << A[j * nr_rows_A + i].x << "+" << A[j * nr_rows_A + i].y << "i" <<" ";
	//			printf("%i,%i: %e\n",i,j,A[j * nr_rows_A + i]);
	//		}
	//		cout << endl;
	//	}
	//	cout << endl;

	for(int i = 0; i < nr_rows_A*nr_cols_A; ++i){
		printf("%i,: %e\n",i,A[i]);
	}
}

static cuComplex * d_weights = NULL;
void update_weights(char * filename){
	char weight_filename[128];
	strcpy(weight_filename, filename);
	FILE * weights;
	float * bf_weights;
	float complex * weights_dc;
	float complex * weights_dc_n;

	// Allocate heap memory for file data
	bf_weights = (float *)malloc(2*N_WEIGHTS*sizeof(float));
	weights_dc = (float complex *)malloc(N_WEIGHTS*sizeof(float complex *));
	weights_dc_n = (float complex *)malloc(N_WEIGHTS*sizeof(float complex *));
	weights = fopen(weight_filename, "r");

	int j;
	if (weights != NULL) {
		fread(bf_weights, sizeof(float), 2*N_WEIGHTS, weights);

		// Convert to complex numbers (do a conjugate at the same time)
		for(j = 0; j < N_WEIGHTS; j++){
			weights_dc_n[j] = bf_weights[2*j] - bf_weights[(2*j)+1]*I;
		}

		// Transpose the weights
		int m,n;
		float complex transpose[N_BEAM][N_ELE*N_BIN];
		for(m=0;m<N_BEAM;m++){
			for(n=0;n<N_ELE*N_BIN;n++){
				transpose[m][n] = weights_dc_n[m*N_ELE*N_BIN + n];
			}
		}
		for(n=0;n<N_ELE*N_BIN;n++){
			for(m=0;m<N_BEAM;m++){
				weights_dc[n*N_BEAM+ m] = transpose[m][n];
			}
		}
		fclose(weights);
	}
	free(bf_weights);

	// Copy weights to device
	cudaMemcpy(d_weights, weights_dc, N_WEIGHTS*sizeof(cuComplex), cudaMemcpyHostToDevice); //r_weights instead of weights_dc //*N_TIME

	free(weights_dc);
}

static cuComplex **d_arr_A = NULL; static cuComplex **d_arr_B = NULL; static cuComplex **d_arr_C = NULL;
static cuComplex * d_beamformed = NULL;
static cuComplex * d_data = NULL;
static cuComplex * d_data1 = NULL;
static float * d_outputs;

void init_beamformer(){
	// Allocate memory for the weights, data, beamformer output, and sti output.

	cudaMalloc((void **)&d_weights, N_WEIGHTS*sizeof(cuComplex)); //*N_TIME

	cudaMalloc((void **)&d_data1, N_SAMP*sizeof(cuComplex));

	cudaMalloc((void **)&d_data, N_SAMP*sizeof(cuComplex));

	cudaError_t err_malloc = cudaMalloc((void **)&d_beamformed, N_TBF*sizeof(cuComplex));
	if (err_malloc != cudaSuccess) {
		printf("CUDA Error (cudaMalloc2): %s\n", cudaGetErrorString(err_malloc));
	}

	cudaMalloc((void **)&d_outputs, N_POL*(N_OUTPUTS*sizeof(float)/2));

	// This is all memory allocated to arrays that are used by gemmBatched.
	// Allocate 3 arrays on CPU
	cudaError_t cudaStat;

	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	nr_rows_A = N_BEAM;
	nr_cols_A = N_ELE;
	nr_rows_B = N_ELE;
	nr_cols_B = N_TIME;
	nr_rows_C = N_BEAM;
	nr_cols_C = N_TIME;

	// Allocate memory to host arrays.
	const cuComplex **h_arr_A = 0; const cuComplex **h_arr_B = 0; cuComplex **h_arr_C = 0;
	h_arr_A = (const cuComplex **)malloc(nr_rows_A * nr_cols_A *N_BIN*sizeof(const cuComplex*));
	h_arr_B = (const cuComplex **)malloc(nr_rows_B * nr_cols_B *N_BIN*sizeof(const cuComplex*));
	h_arr_C = (cuComplex **)malloc(nr_rows_C * nr_cols_C *N_BIN*sizeof(cuComplex*));

	// Allocate memory for each batch in an array.
	for(int i = 0; i < N_BIN; i++){
		h_arr_A[i] = d_weights + i*nr_rows_A*nr_cols_A;
		h_arr_B[i] = d_data + i*nr_rows_B*nr_cols_B;
		h_arr_C[i] = d_beamformed + i*nr_rows_C*nr_cols_C;
	}

	//	delete[] d_A;
	//	delete[] d_B;

	// Allocate memory to arrays on device.
	cudaStat = cudaMalloc((void **)&d_arr_A,nr_rows_A * nr_cols_A * N_BIN * sizeof(cuComplex*));
	assert(!cudaStat);
	cudaStat = cudaMalloc((void **)&d_arr_B,nr_rows_B * nr_cols_B * N_BIN * sizeof(cuComplex*));
	assert(!cudaStat);
	cudaStat = cudaMalloc((void **)&d_arr_C,nr_rows_C * nr_cols_C * N_BIN * sizeof(cuComplex*));
	assert(!cudaStat);

	// Copy memory from host to device.
	cudaStat = cudaMemcpy(d_arr_A,h_arr_A,nr_rows_A * nr_cols_A * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
	assert(!cudaStat);
	cudaStat = cudaMemcpy(d_arr_B,h_arr_B,nr_rows_B * nr_cols_B * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
	assert(!cudaStat);
	cudaStat = cudaMemcpy(d_arr_C,h_arr_C,nr_rows_C * nr_cols_C * N_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
	assert(!cudaStat);

}

__global__
void data_restructure(cuComplex * data, cuComplex * data_restruc){

	int e = threadIdx.x;
	int t = blockIdx.x;
	int f = blockIdx.y;

	//Restructure data so that the frequency bin is the slowest moving index
	data_restruc[f*N_TIME*N_ELE + t*N_ELE + e] = data[t*N_BIN*N_ELE + f*N_ELE + e];
}

void data_in(char * input_filename){
	FILE * data;

	// File data pointers
	float * bf_data;

	// Complex data pointers
	float complex * data_dc;

	// Allocate heap memory for file data
	bf_data = (float *)malloc(2*N_SAMP*sizeof(float));
	data_dc = (float complex *)malloc(N_SAMP*sizeof(float complex *));

	// Open files
	data = fopen(input_filename, "r");

	/*********************************************************
	 * Read in Data
	 *********************************************************/
	if (data != NULL) {
		fread(bf_data, sizeof(float), 2*N_SAMP, data);
		int j;
		// Make 'em complex!
		for (j = 0; j < N_SAMP; j++) {
			data_dc[j] = bf_data[2*j] + bf_data[(2*j)+1]*I;
		}

		// Specify grid and block dimensions
		dim3 dimBlock_d(N_ELE, 1, 1);
		dim3 dimGrid_d(N_TIME, N_BIN, 1);

		cuComplex * d_data_in = d_data1;
		cuComplex * d_data_out = d_data;

		cudaMemcpy(d_data_in,    data_dc,   N_SAMP*sizeof(cuComplex), cudaMemcpyHostToDevice);

		// Restructure data for cublasCgemmBatched function.
		data_restructure<<<dimGrid_d, dimBlock_d>>>(d_data_in, d_data_out);

		fclose(data);
	}
	free(bf_data);
	free(data_dc);
}

void beamform(cublasHandle_t handle) {
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C;
	nr_rows_A = N_BEAM;
	nr_cols_A = N_ELE;
	nr_rows_B = N_ELE;
	nr_cols_B = N_TIME;
	nr_rows_C = N_BEAM;

	// Leading dimensions are always the rows of each matrix since the data is stored in a column-wise order.
	int lda=nr_rows_A,ldb=nr_rows_B,ldc=nr_rows_C;
	cuComplex alf;
	cuComplex bet;

	alf.x = 1;
	alf.y = 0;
	bet.x = 0;
	bet.y = 0;

	int batchCount = N_BIN; 				// There must be the same number of batches in each array.

	cublasStatus_t stat;
	/*
		This function performs a matrix multiplication of the data and the weights.
		Weights - d_arr_A, Data - d_arr_B, and the output - d_arr_C.
	*/
	stat = cublasCgemmBatched(
			handle,							// handle to the cuBLAS library context.
			CUBLAS_OP_N,					// Operation on matrices within array A.
			CUBLAS_OP_N,					// Operation on matrices within array B.
			nr_rows_A,						// Number of rows in matrix A and C.
			nr_cols_B,						// Number of columns in matrix B and C.
			nr_cols_A,						// Number of columns and rows in matrix A and B respectively.
			&alf,							// Scalar used for multiplication.
			(const cuComplex **)d_arr_A,	// Weight array of pointers.
			lda,							// Leading dimension of each batch or matrix in array A.
			(const cuComplex **)d_arr_B,	// Data array of pointers.
			ldb,							// Leading dimension of each batch or matrix in array B.
			&bet,							// Scalar used for multiplication.
			(cuComplex **)d_arr_C,			// Output array of pointers.
			ldc,							// Leading dimension of each batch or matrix in array C.
			batchCount);					// Number of batches in each array.


	if(stat != CUBLAS_STATUS_SUCCESS){
		cerr << "cublasCgemmBatched failed" << endl;
		exit(1);
	}
	assert(!cudaGetLastError());

	//Free GPU memory
	//	cudaFree(d_A);
	//	cudaFree(d_B);
	//	cudaFree(d_C);

	// Destroy the handle
	//cublasDestroy(handle);

}

__global__
void sti_reduction(cuComplex * data_in, float * data_out) {

	int f = blockIdx.x;
	int b = blockIdx.y;
	int t = threadIdx.x;
	int s = blockIdx.z;

	int h = sample_idx(s*N_TIME_STI + t,b,f);						// Preprocessor macro used for the output of the beamformer. More detail can be seen in the header file. (First set of beams)
	int h1 = sample_idx(s*N_TIME_STI + t,b+N_BEAM1,f);				// Preprocessor macro used for the output of the beamformer. More detail can be seen in the header file. (Last set of beams)

	// Temporary variables used for updating.
	float beam_power1;
	float beam_power2;
	float cross_power1;
	float cross_power2;

	cuFloatComplex samp1;
	cuFloatComplex samp2;
	float scale = 1.0/N_TIME_STI; 									// Scale power by number of samples per STI window.

	__shared__ cuFloatComplex reduced_array1[N_STI_BLOC];
	__shared__ cuFloatComplex reduced_array[N_STI_BLOC];

	if (t < N_TIME_STI) {
		// X polarization (XX*).
		samp1.x = data_in[h].x;
		samp1.y = data_in[h].y;
		beam_power1 = (samp1.x * samp1.x) + (samp1.y * samp1.y);	// Beamformer output multiplied by its conjugate (absolute value squared).
		reduced_array[t].x = beam_power1;

		// Y polarization (YY*).
		samp2.x = data_in[h1].x;
		samp2.y = data_in[h1].y;
		beam_power2 = (samp2.x * samp2.x) + (samp2.y * samp2.y);	// Beamformer output multiplied by its conjugate (absolute value squared).
		reduced_array[t].y = beam_power2;

		// Cross polarization (XY*).
		cross_power1 = (samp1.x * samp2.x) + (samp1.y * samp2.y);	// Real part of cross polarization.
		cross_power2 = (samp1.y * samp2.x) - (samp1.x * samp2.y);	// Imaginary part of cross polarization.
		reduced_array1[t].x = cross_power1;
		reduced_array1[t].y = cross_power2;
	}
	else{
		reduced_array[t].x = 0.0;
		reduced_array[t].y = 0.0;
		reduced_array1[t].x = 0.0;
		reduced_array1[t].y = 0.0;
	}
	__syncthreads();

	// Reduction is performed by splitting up the threads in each block and summing them all up.
	// The number of threads in each block needs to be a power of two in order for the reduction to work. (No left over threads).
	for(int k = blockDim.x/2; k>0; k>>=1){
		if(t<k){
			reduced_array[t].x += reduced_array[t+k].x;
			reduced_array[t].y += reduced_array[t+k].y;
			reduced_array1[t].x += reduced_array1[t+k].x;
			reduced_array1[t].y += reduced_array1[t+k].y;
		}
		__syncthreads();
	}

	// After reduction is complete, assign each reduced to value to appropriate position in output array.
	if(t == 0){
		data_out[output_idx(0,b,s,f)] = reduced_array[0].x*scale; 	// XX*.
		data_out[output_idx(1,b,s,f)] = reduced_array[0].y*scale; 	// YY*.
		data_out[output_idx(2,b,s,f)] = reduced_array1[0].x*scale; 	// XY* real.
		data_out[output_idx(3,b,s,f)] = reduced_array1[0].y*scale;	// XY* imaginary.
	}
}

void run_beamformer(cublasHandle_t handle, float * data_out){
	// Specify grid and block dimensions
	dim3 dimBlock(N_STI_BLOC, 1, 1);
	dim3 dimGrid(N_BIN, N_BEAM1, N_STI);

	printf("Starting beamformer\n");

	// Call beamformer function containing cublasCgemmBatched()
	beamform(handle);
	cudaError_t err_code = cudaGetLastError();
	if (err_code != cudaSuccess) {
		printf("CUDA Error (beamform): %s\n", cudaGetErrorString(err_code));
	}

	cuComplex * d_data_in = d_beamformed;
	float * d_data_out = d_outputs;

	printf("Starting sti_reduction\n");

	// Call STI reduction kernel.
	sti_reduction<<<dimGrid, dimBlock>>>(d_data_in, d_data_out);

	printf("Finishing sti_reduction\n");

	err_code = cudaGetLastError();
	if (err_code != cudaSuccess) {
		printf("CUDA Error (sti_reduction): %s\n", cudaGetErrorString(err_code));
	}

	// Copy output data from device to host.
	cudaMemcpy(data_out, d_data_out, N_POL*(N_OUTPUTS*sizeof(float)/2),cudaMemcpyDeviceToHost);

	cudaFree(d_data);
	cudaFree(d_outputs);
}
