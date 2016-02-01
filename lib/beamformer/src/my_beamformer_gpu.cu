#include "beamformer_gpu.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>

void printUsage();

int main(int argc, char * argv[]) {
	// Parse input
	if (argc != 4) {
		printUsage();
		return -1;
	}
	char input_filename[128];
	char weight_filename[128];
	char output_filename[128];

	strcpy(input_filename,  argv[1]);
	strcpy(weight_filename, argv[2]);
	strcpy(output_filename, argv[3]);

	// File pointers
	FILE * data;
	FILE * weights;
	
	// File data pointers
	float * bf_data;
	float * bf_weights;

	// Complex data pointers
	float complex * data_dc;
	float complex * weights_dc;

	// Allocate heap memory for file data
	bf_data = (float *)malloc(2*N_SAMP*sizeof(float));
	bf_weights = (float *)malloc(2*N_WEIGHTS*sizeof(float));
	data_dc = (float complex *)malloc(N_SAMP*sizeof(float complex *));
	weights_dc = (float complex *)malloc(N_WEIGHTS*sizeof(float complex *));

	// Open files
	data = fopen(input_filename, "r");
	weights = fopen(weight_filename, "r");

	// Read in data
	int j;
	if (data != NULL) {
		fread(bf_data, sizeof(float), 2*N_SAMP, data);

		// Make 'em complex!
		for (j = 0; j < N_SAMP; j++) {
			data_dc[j] = bf_data[2*j] + bf_data[(2*j)+1]*I;
		}
		fclose(data);
	}
	free(bf_data);

	if (weights != NULL) {
		fread(bf_weights, sizeof(float), 2*N_WEIGHTS, weights);
		// Make 'em complex!
		for (j = 0; j < N_WEIGHTS; j++) {
			weights_dc[j] = bf_weights[2*j] + bf_weights[(2*j)+1]*I;
		}
		fclose(weights);
	}
	free(bf_weights);

	// Allocate memory for the output
	float * output_f;
	output_f = (float *)calloc(N_OUTPUTS,sizeof(float));

	struct timespec tstart = {0,0};
	struct timespec tstop  = {0,0};
	clock_gettime(CLOCK_MONOTONIC, &tstart);

	// Specify grid and block dimensions
	dim3 dimBlock(N_STI_BLOC, 1, 1);
	dim3 dimGrid(N_BIN, N_BEAM, N_STI);
	
	dim3 dimBlock2(N_ELE_BLOC, 1, 1);
	dim3 dimGrid2(N_TIME, N_BIN, N_BEAM);

	cuFloatComplex * d_data;
	cuFloatComplex * d_weights;
	cuFloatComplex * d_beamformed;//////////
	float * d_outputs;

	//cudaMalloc((void **)&d_data, N_SAMP*sizeof(cuDoubleComplex));
	//cudaMalloc((void **)&d_weights, N_WEIGHTS*sizeof(cuDoubleComplex));
	//cudaMalloc((void **)&d_outputs, N_OUTPUTS*sizeof(float));
	cudaError_t err_malloc = cudaMalloc((void **)&d_data, (N_SAMP + N_WEIGHTS)*sizeof(cuFloatComplex) + N_OUTPUTS*sizeof(float));
	if (err_malloc != cudaSuccess) {
		printf("CUDA Error (cudaMalloc1): %s\n", cudaGetErrorString(err_malloc));
	}
	err_malloc = cudaMalloc((void **)&d_beamformed, N_TBF*sizeof(cuFloatComplex));
	if (err_malloc != cudaSuccess) {
		printf("CUDA Error (cudaMalloc2): %s\n", cudaGetErrorString(err_malloc));
	}

	d_weights = d_data + N_SAMP;
	d_outputs = (float *)(d_data + N_SAMP + N_WEIGHTS);
	cudaMemset(d_outputs, 0.0, N_OUTPUTS*sizeof(float));
	
	//printf("data_dc weights_dc %.7e %e\n",data_dc,weights_dc);
	cudaMemcpy(d_data,    data_dc,   N_SAMP*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(d_weights, weights_dc, N_WEIGHTS*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);


	//printf("data_dc:\t%.7e+%.7e*I\n weights_dc:\t%.7e+%.7e*I\n",data_dc[0],weights_dc[0]);

	// Run the beamformer
	//printf("D_data D_weights %.7e + %.7e*I\n",temp);
	printf("Starting beamformer\n");
	beamform<<<dimGrid2, dimBlock2>>>(d_data, d_weights, d_beamformed);//beamform<<<dimGrid, dimBlock>>>(d_data, d_weights, d_beamformed);
	//printf("D_data D_weights D_outputs %.7e %e %e\n",d_data,d_weights,d_outputs);
	printf("Finishing beamformer\n");

	cudaError_t err_code = cudaGetLastError();
	if (err_code != cudaSuccess) {
		printf("CUDA Error (beamform): %s\n", cudaGetErrorString(err_code));
	}

	//printf("Beamformed %e+%e*I\n", temp);

	printf("Starting sti_reduction\n");
	sti_reduction<<<dimGrid, dimBlock>>>(d_beamformed,d_outputs);
	printf("Finishing sti_reduction\n");

	err_code = cudaGetLastError();
	if (err_code != cudaSuccess) {
		printf("CUDA Error (sti_reduction): %s\n", cudaGetErrorString(err_code));
	}
	
	
	cudaMemcpy(output_f, d_outputs, N_OUTPUTS*sizeof(float), cudaMemcpyDeviceToHost);
	//printf("Output %e\n",output_f[0]);
	cudaFree(d_data);
	cudaFree(d_weights);
	cudaFree(d_outputs);

	clock_gettime(CLOCK_MONOTONIC, &tstop);
	//printf("Beamformer elapsed time: %.5f seconds\n",
		//((double)tstop.tv_sec + 1.0e-9*tstop.tv_nsec) -
		//((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
	
	// Save output data to file
	FILE * output;
	output = fopen(output_filename, "w");
	fwrite(output_f, sizeof(float), N_OUTPUTS, output);
	fclose(output);

	free(data_dc);
	free(weights_dc);
	free(output_f);

	return 0;
}

void printUsage() {
	printf("Usage: my_beamformer <input_filename> <weight_filename> <output_filename>\n");
}

//For makefile at the very end "-fno-exceptions -fno-rtti"
