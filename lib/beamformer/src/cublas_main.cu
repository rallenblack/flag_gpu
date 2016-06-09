#include "cublas_beamformer.h"

#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <curand.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

using namespace std;

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

	/*********************************************************
	 *Create a handle for CUBLAS
	 *********************************************************/
	cublasHandle_t handle;
	cublasCreate(&handle);

	/*********************************************************
	 * Initialize beamformer
	 *********************************************************/
	init_beamformer();

	/*********************************************************
	 * Update in weights
	 *********************************************************/
	update_weights(weight_filename);

	/*********************************************************
	 * Input data and restructure for cublasCgemmBatched()
	 *********************************************************/
	data_in(input_filename);

	// Allocate memory for the output
	float * output_f;
	output_f = (float *)calloc(N_POL*(N_OUTPUTS/2),sizeof(float));

	/*********************************************************
     * Run beamformer
     *********************************************************/
	run_beamformer(handle, output_f);

	// Save output data to file
	FILE * output;
	output = fopen(output_filename, "w");
	fwrite(output_f, sizeof(float), N_POL*(N_OUTPUTS/2), output);
	fclose(output);

	free(output_f);
	//	cublasDestroy(handle);

	return 0;
}

void printUsage() {
	printf("Usage: my_beamformer <input_filename> <weight_filename> <output_filename>\n");
}
//  // Start and stop time - Used time certain sections of code (Not very accurate, use profiler or cudaThreadSynchronize())
//	struct timespec tstart = {0,0};
//	struct timespec tstop  = {0,0};
//	clock_gettime(CLOCK_MONOTONIC, &tstart);

//	clock_gettime(CLOCK_MONOTONIC, &tstop);
//	printf("Data and Weights restructure elapsed time: %.5f seconds\n",
//	((double)tstop.tv_sec + 1.0e-9*tstop.tv_nsec) -
//	((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));

//For makefile at the very end "-fno-exceptions -fno-rtti"
