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
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_beamformer.h"

using namespace std;

void beamform();

__global__
void transpose(signed char* data, signed char* tra_data);

__global__
void data_restructure(signed char * data, cuComplex * data_restruc);

__global__
void sti_reduction(cuComplex * data_in, float * data_out);


// Struct defintion for beamformer metadata
typedef struct bf_metadata_struct {
	float offsets[14];
	char cal_filename[65];
	char algorithm[65];
	char weight_filename[65];
	long long unsigned int xid;
} bf_metadata;

static bf_metadata my_metadata;


static cuComplex * d_weights = NULL;

void update_weights(char * filename){

	printf("RTBF: In update_weights()...\n");

	char weight_filename[128];
	strcpy(weight_filename, filename);
	FILE * weights;
	float * bf_weights;
	float complex * weights_dc;
	float complex * weights_dc_n;

	// Allocate heap memory for file data
	bf_weights = (float *)malloc(2*BN_WEIGHTS*sizeof(float));
	weights_dc = (float complex *)malloc(BN_WEIGHTS*sizeof(float complex *));
	weights_dc_n = (float complex *)malloc(BN_WEIGHTS*sizeof(float complex *));

	// open weight file
	weights = fopen(weight_filename, "r");

	int j;
	if (weights != NULL) {

		fread(bf_weights, sizeof(float), 2*BN_WEIGHTS, weights);

		fread(my_metadata.offsets, sizeof(float), 14, weights);
		fread(my_metadata.cal_filename, sizeof(char), 64, weights);
		fread(my_metadata.algorithm, sizeof(char), 64, weights);
		fread(&(my_metadata.xid), sizeof(long long unsigned int), 1, weights);

		my_metadata.cal_filename[64] = '\0';
		my_metadata.algorithm[64] = '\0';


		// Extract all path information from weight_filename for metadata
		char * short_filename = strrchr(weight_filename, '/');
		if (short_filename != NULL) {
			strcpy(my_metadata.weight_filename, short_filename+1);
		}
		else {
			strcpy(my_metadata.weight_filename, weight_filename);
		}

		// Convert to complex numbers (do a conjugate at the same time)
		for(j = 0; j < BN_WEIGHTS; j++){
			weights_dc_n[j] = bf_weights[2*j] - bf_weights[(2*j)+1]*I;
		}

		// Transpose the weights
		int m,n;
		float complex transpose[BN_BEAM][BN_ELE_BLOC*BN_BIN];
		for(m=0;m<BN_BEAM;m++){
			for(n=0;n<BN_ELE_BLOC*BN_BIN;n++){
				transpose[m][n] = weights_dc_n[m*BN_ELE_BLOC*BN_BIN + n];
			}
		}
		for(n=0;n<BN_ELE_BLOC*BN_BIN;n++){
			for(m=0;m<BN_BEAM;m++){
				weights_dc[n*BN_BEAM+ m] = transpose[m][n];
			}
		}
		fclose(weights);
	}

	// Copy weights to device
	cudaMemcpy(d_weights, weights_dc, BN_WEIGHTS*sizeof(cuComplex), cudaMemcpyHostToDevice); //r_weights instead of weights_dc //*BN_TIME

	// free memory
	free(weights_dc);
	free(weights_dc_n);
	free(bf_weights);

	return; 
}

void bf_get_offsets(float * offsets){
	for(int i = 0; i<BN_BEAM; i++){
		offsets[i] = my_metadata.offsets[i];
	}
}

void bf_get_cal_filename(char * cal_filename){
	for(int i = 0; i< 65; i++){
		cal_filename[i] = my_metadata.cal_filename[i];
	}
}

void bf_get_algorithm(char * algorithm){
	for(int i = 0; i< 65; i++){
		algorithm[i] = my_metadata.algorithm[i];
	}
}

void bf_get_weight_filename(char * weight_filename){
	int num_chars = strlen(my_metadata.weight_filename);
	for (int i = 0; i < num_chars; i++) {
		weight_filename[i] = my_metadata.weight_filename[i];
	}
	for (int i = num_chars; i < 64; i++) {
		weight_filename[i] = ' ';
	}
	weight_filename[64] = '\0';
}

long long unsigned int bf_get_xid(){
	return my_metadata.xid;
}

static cuComplex * d_beamformed = NULL;
static cuComplex * d_data = NULL;
static signed char * d_data1 = NULL; // Device memory for input data
static signed char * d_data2 = NULL;
static float * d_outputs;

static cublasHandle_t handle;
static cuComplex **d_arr_A = NULL;
static cuComplex **d_arr_B = NULL;
static cuComplex **d_arr_C = NULL;
void init_beamformer(){

	// Allocate memory for the weights, data, beamformer output, and sti output.
	cudaMalloc((void **)&d_weights, BN_WEIGHTS*sizeof(cuComplex)); //*BN_TIME

	cudaMalloc((void **)&d_data1, 2*BN_SAMP*sizeof(signed char));

	//cudaMalloc((void **)&d_data2, 2*BN_SAMP*sizeof(signed char));

	cudaMalloc((void **)&d_data, BN_SAMP*sizeof(cuComplex));

	cudaError_t err_malloc = cudaMalloc((void **)&d_beamformed, BN_TBF*sizeof(cuComplex));
	if (err_malloc != cudaSuccess) {
		printf("CUDA Error (cudaMalloc2): %s\n", cudaGetErrorString(err_malloc));
	}

	cudaMalloc((void **)&d_outputs, BN_POL*(BN_OUTPUTS*sizeof(float)/2));

    /**********************************************************
    * Create a handle for CUBLAS
    **********************************************************/
    cublasCreate(&handle);
	cudaError_t cudaStat;

	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

	nr_rows_A = BN_BEAM;
	nr_cols_A = BN_ELE_BLOC;
	nr_rows_B = BN_ELE_BLOC;
	nr_cols_B = BN_TIME;
	nr_rows_C = BN_BEAM;
	nr_cols_C = BN_TIME;

	// Allocate memory to host arrays - This is all memory allocated to arrays that are used by gemmBatched. Allocate 3 arrays on CPU
	const cuComplex **h_arr_A = 0;
	const cuComplex **h_arr_B = 0;
	cuComplex **h_arr_C = 0;

	h_arr_A = (const cuComplex **)malloc(nr_rows_A * nr_cols_A *BN_BIN*sizeof(const cuComplex*));
	h_arr_B = (const cuComplex **)malloc(nr_rows_B * nr_cols_B *BN_BIN*sizeof(const cuComplex*));
	h_arr_C = (cuComplex **)malloc(nr_rows_C * nr_cols_C *BN_BIN*sizeof(cuComplex*));

	// Allocate memory for each batch in an array.
	for(int i = 0; i < BN_BIN; i++){
		h_arr_A[i] = d_weights + i*nr_rows_A*nr_cols_A;
		h_arr_B[i] = d_data + i*nr_rows_B*nr_cols_B;
		h_arr_C[i] = d_beamformed + i*nr_rows_C*nr_cols_C;
	}

	// Allocate memory to arrays on device.
	cudaStat = cudaMalloc((void **)&d_arr_A,nr_rows_A * nr_cols_A * BN_BIN * sizeof(cuComplex*));
	assert(!cudaStat);
	cudaStat = cudaMalloc((void **)&d_arr_B,nr_rows_B * nr_cols_B * BN_BIN * sizeof(cuComplex*));
	assert(!cudaStat);
	cudaStat = cudaMalloc((void **)&d_arr_C,nr_rows_C * nr_cols_C * BN_BIN * sizeof(cuComplex*));
	assert(!cudaStat);

	// Copy memory from host to device.
	cudaStat = cudaMemcpy(d_arr_A,h_arr_A,nr_rows_A * nr_cols_A * BN_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
	assert(!cudaStat);
	cudaStat = cudaMemcpy(d_arr_B,h_arr_B,nr_rows_B * nr_cols_B * BN_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
	assert(!cudaStat);
	cudaStat = cudaMemcpy(d_arr_C,h_arr_C,nr_rows_C * nr_cols_C * BN_BIN * sizeof(cuComplex*),cudaMemcpyHostToDevice);
	assert(!cudaStat);

	free(h_arr_A);
	free(h_arr_B);
	free(h_arr_C);

	return;
        
}

signed char * data_in(char * input_filename){
	FILE * data;

	// File data pointers
	signed char * bf_data;

	// Complex data pointers
	// float complex * data_dc;

	// Allocate heap memory for file data
	bf_data = (signed char *)malloc(2*BN_SAMP*sizeof(signed char));
	//data_dc = (float complex *)malloc(BN_SAMP*sizeof(float complex *));

	// Open files
	data = fopen(input_filename, "r");

	/*********************************************************
	 * Read in Data
	 *********************************************************/
	if (data != NULL) {
		fread(bf_data, sizeof(signed char), 2*BN_SAMP, data);
		/*
		int j;
		// Make 'em complex!
		for (j = 0; j < BN_SAMP; j++) {
			data_dc[j] = bf_data[2*j] + bf_data[(2*j)+1]*I;
		}
		*/

		// Specify grid and block dimensions
		// dim3 dimBlock_d(BN_ELE, 1, 1);
		// dim3 dimGrid_d(BN_TIME, BN_BIN, 1);

		//cuComplex * d_data_in = d_data1;
		//cuComplex * d_data_out = d_data;

		//cudaMemcpy(d_data_in,    data_dc,   BN_SAMP*sizeof(cuComplex), cudaMemcpyHostToDevice);

		// Restructure data for cublasCgemmBatched function.
		//data_restructure<<<dimGrid_d, dimBlock_d>>>(d_data_in, d_data_out);

		fclose(data);
	}
	//free(bf_data);
	//free(data_dc);
	return bf_data;
}

void beamform() {
	int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C;
	nr_rows_A = BN_BEAM;
	nr_cols_A = BN_ELE_BLOC;
	nr_rows_B = BN_ELE_BLOC;
	nr_cols_B = BN_TIME;
	nr_rows_C = BN_BEAM;

	// Leading dimensions are always the rows of each matrix since the data is stored in a column-wise order.
	int lda=nr_rows_A, ldb=nr_rows_B, ldc=nr_rows_C;
	cuComplex alf;
	cuComplex bet;

	alf.x = 1;
	alf.y = 0;
	bet.x = 0;
	bet.y = 0;

	int batchCount = BN_BIN; 				// There must be the same number of batches in each array.

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

	if (stat == CUBLAS_STATUS_INVALID_VALUE) {
		printf("RTBF: Invalid CUBLAS values\n");
	} else if (stat == CUBLAS_STATUS_EXECUTION_FAILED) {
		printf("RTBF: Execution failed.\n");
	}

	if(stat != CUBLAS_STATUS_SUCCESS){
		cerr << "cublasCgemmBatched failed" << endl;
		exit(1);
	}
	assert(!cudaGetLastError());

}

__global__ 
void transpose(signed char* data, signed char* tra_data) {
	int i = threadIdx.x; 
	int c = threadIdx.y;

	int m = blockIdx.x;
	int f = blockIdx.y;
	int t = blockIdx.z;

	//int Nm = gridDim.x; // number of mcnts (packets)
	int Nf = gridDim.y; // number of f-engines (ROACHES)
	int Nt = gridDim.z; // time samples per mcnt

	int Ni = blockDim.x; // inputs per f-engine (aka antenna elements per ROACH)
	int Nc = blockDim.y; // bins per mcnt

	int in_idx  = i + Ni*c + Nc*Ni*t + Nt*Nc*Ni*f + Nf*Nt*Nc*Ni*m;	
	int out_idx = i + Ni*f + Nf*Ni*c + Nc*Nf*Ni*t + Nt*Nc*Nf*Ni*m;

	tra_data[2*out_idx] = data[2*in_idx];
	tra_data[2*out_idx + 1] = data[2*in_idx+1];

	return;
}

__global__
void data_restructure(signed char * data, cuComplex * data_restruc){

	/*
		Repurpose the transpose thread in the hashpipe codes by performing the transpose in the GPU.
		The motivation was, why transpose then transpose again? Why not just perform one transpose
		in the GPU which would be faster anyway.
	*/

        /*
	int i = threadIdx.x; 
	int c = threadIdx.y;

	int m = blockIdx.x;
	int f = blockIdx.y;
	int t = blockIdx.z;

	int Nm = gridDim.x; // number of mcnts (packets)
	int Nf = gridDim.y; // number of f-engines (ROACHES)
	int Nt = gridDim.z; // time samples per mcnt

	int Ni = blockDim.x; // inputs per f-engine (aka antenna elements per ROACH)
	int Nc = blockDim.y; // bins per mcnt

	int in_idx  = i + Ni*c + Nc*Ni*t + Nt*Nc*Ni*f + Nf*Nt*Nc*Ni*m;
	int out_idx = i + Ni*f + Nf*Ni*t + Nt*Nf*Ni*m + Nm*Nt*Nf*Ni*c;

	data_restruc[out_idx].x = data[2*in_idx]*1.0f;
	data_restruc[out_idx].y = data[2*in_idx + 1]*1.0f;

	return;
        */
	
	// Original Code
	int e = threadIdx.x;
	int t = blockIdx.x;
	int f = blockIdx.y;

	//Restructure data so that the frequency bin is the slowest moving index
	data_restruc[f*BN_TIME*BN_ELE_BLOC + t*BN_ELE_BLOC + e].x = data[2*(t*BN_BIN*BN_ELE_BLOC + f*BN_ELE_BLOC + e)]*1.0f;
	data_restruc[f*BN_TIME*BN_ELE_BLOC + t*BN_ELE_BLOC + e].y = data[2*(t*BN_BIN*BN_ELE_BLOC + f*BN_ELE_BLOC + e) + 1]*1.0f;
	

	return;

}


__global__
void sti_reduction(cuComplex * data_in, float * data_out) {

	int f = blockIdx.x;
	int b = blockIdx.y;
	int t = threadIdx.x;
	int s = blockIdx.z;

	int h = sample_idx(s*BN_TIME_STI + t,b,f);						// Preprocessor macro used for the output of the beamformer. More detail can be seen in the header file. (First set of beams)
	int h1 = sample_idx(s*BN_TIME_STI + t,b+BN_BEAM1,f);			// Preprocessor macro used for the output of the beamformer. More detail can be seen in the header file. (Last set of beams)

	// Temporary variables used for updating.
	float beam_power1;
	float beam_power2;
	float cross_power1;
	float cross_power2;

	cuFloatComplex samp1;
	cuFloatComplex samp2;
	float scale = 1.0/BN_TIME_STI; 									// Scale power by number of samples per STI window.

	__shared__ cuFloatComplex reduced_array1[BN_STI_BLOC];
	__shared__ cuFloatComplex reduced_array[BN_STI_BLOC];

	if (t < BN_TIME_STI) {
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

	return;
}

void run_beamformer(signed char * data_in, float * data_out) {

	cudaError_t err_code;
	// Specify grid and block dimensions
	dim3 dimBlock(BN_STI_BLOC, 1, 1);
	dim3 dimGrid(BN_BIN, BN_BEAM1, BN_STI);

	// Specify grid and block dimensions
	dim3 dimBlock_d(BN_ELE_BLOC, 1, 1);
	dim3 dimGrid_d(BN_TIME, BN_BIN, 1);

	int Nm = 200;
	int Nf = 8;
	int Nt = 20;
	int Nc = 25;
	int Ni = 8;
	dim3 gridDim_transpose(Nm, Nf, Nt);
	dim3 blockDim_transpose(Ni, Nc, 1);

	//signed char* d_tra_data_in = d_data1;
	//signed char* d_tra_data_out = d_data2;
	signed char * d_restruct_in = d_data1;
	cuComplex * d_restruct_out = d_data;

	cudaMemcpy(d_restruct_in, data_in, 2*BN_SAMP*sizeof(signed char), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_tra_data_in, data_in, 2*BN_SAMP*sizeof(signed char), cudaMemcpyHostToDevice);
	err_code = cudaGetLastError();
	if (err_code != cudaSuccess) {
		printf("RTBF: cudaMemcpy Failed: %s\n", cudaGetErrorString(err_code));
	}

	// Transpose the data
	// transpose<<<gridDim_transpose, blockDim_transpose>>>(d_tra_data_in, d_tra_data_out);
	// if (err_code != cudaSuccess) {
	// 	printf("RTBF: CUDA Error (transpose): %s\n", cudaGetErrorString(err_code));
	// }

	// Restructure data for cublasCgemmBatched function.
	//data_restructure<<<gridDim_transpose, blockDim_transpose>>>(d_tra_data_in, d_restruct_out);
	//data_restructure<<<gridDim_transpose, blockDim_transpose>>>(d_restruct_in, d_restruct_out);
	data_restructure<<<dimGrid_d, dimBlock_d>>>(d_restruct_in, d_restruct_out);
	if (err_code != cudaSuccess) {
		printf("RTBF: CUDA Error (data_restructure): %s\n", cudaGetErrorString(err_code));
	}

	// Call beamformer function containing cublasCgemmBatched()
	beamform();
	err_code = cudaGetLastError();
	if (err_code != cudaSuccess) {
		printf("CUDA Error (beamform): %s\n", cudaGetErrorString(err_code));
	}

	cuComplex * d_sti_in = d_beamformed;
	float * d_sti_out = d_outputs;

	// Call STI reduction kernel.
	sti_reduction<<<dimGrid, dimBlock>>>(d_sti_in, d_sti_out);

	err_code = cudaGetLastError();
	if (err_code != cudaSuccess) {
		printf("CUDA Error (sti_reduction): %s\n", cudaGetErrorString(err_code));
	}

	// Copy output data from device to host.
	cudaMemcpy(data_out, d_sti_out, BN_POL*(BN_OUTPUTS*sizeof(float)/2),cudaMemcpyDeviceToHost);

	return;
}


void rtbfCleanup() {
	// Free up GPU memory at the end of a program
	if (d_beamformed != NULL) {
		cudaFree(d_beamformed);
	}

	if (d_data != NULL) {
		cudaFree(d_data);
	}

	if (d_data1 != NULL) {
		cudaFree(d_data1);
	}

	if (d_data2 != NULL) {
		cudaFree(d_data2);
	}

	if (d_outputs != NULL) {
		cudaFree(d_outputs);
	}

	if (d_weights != NULL) {
		cudaFree(d_weights);
	}

	if (d_arr_A != NULL) {
		cudaFree(d_arr_A);
	}

	if (d_arr_B != NULL) {
		cudaFree(d_arr_B);
	}

	if (d_arr_C != NULL) {
		cudaFree(d_arr_C);
	}
	// Free up and release cublas handle
	cublasDestroy(handle);

}
