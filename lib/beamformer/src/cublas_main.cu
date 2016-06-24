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

// Main-specific function prototypes
signed char * data_in(char * input_filename);
void printUsage();

int main(int argc, char * argv[]) {
	printf("Starting up the main process!\n");
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
	 * Initialize beamformer
	 *********************************************************/
	init_beamformer();

	/*********************************************************
	 * Update in weights
	 *********************************************************/
	update_weights(weight_filename);

	float offsets[BN_BEAM];
	char cal_filename[65];
	char algorithm[65];
	char weight_file[65];
	long long unsigned int xid;

	bf_get_offsets(offsets);
	bf_get_cal_filename(cal_filename);
	bf_get_algortihm(algorithm);
	bf_get_weight_filename(weight_file);
	xid = bf_get_xid();

	for (int i = 0; i < 7; i++) {
		printf("Offset %d = (%f, %f)\n", i, offsets[2*i], offsets[2*i+1]);
	}
	printf("Calibration Filename = %s\n", cal_filename);
	printf("Algorithm for Weights = %s\n", algorithm);
	printf("XID = %llu\n", xid);
	printf("Weight filename = %s\n", weight_file);

	/*********************************************************
	 * Input data and restructure for cublasCgemmBatched()
	 *********************************************************/
	signed char * input_f = data_in(input_filename);

	// Allocate memory for the output
	float * output_f;
	output_f = (float *)calloc(BN_POL*(BN_OUTPUTS/2),sizeof(float));

	/*********************************************************
         * Run beamformer
         *********************************************************/
	run_beamformer(input_f, output_f);

	// Save output data to file
	FILE * output;
	output = fopen(output_filename, "w");
	fwrite(output_f, sizeof(float), BN_POL*(BN_OUTPUTS/2), output);
	fclose(output);

	free(output_f);

	return 0;
}

void printUsage() {
	printf("Usage: cublas_main <input_filename> <weight_filename> <output_filename>\n");
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
