#include "flag_beamformer.h"
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
	
	// File data pointers
	float * bf_data;
	unsigned char * bf_data_char;

	// Allocate heap memory for file data
	bf_data = (float *)malloc(2*N_SAMP*sizeof(float));
	bf_data_char = (unsigned char *)malloc(2*N_SAMP*sizeof(float));

	// Open files
	data = fopen(input_filename, "r");

	// Read in data
	if (data != NULL) {
		fread(bf_data, sizeof(float), 2*N_SAMP, data);
		fclose(data);
	}

	float min = bf_data[0];
	float max = bf_data[0];
	for (int j = 0; j < 2*N_SAMP; j++) {
		//bf_data_char[j] = bf_data[j];
		if (bf_data[j] < min) min = bf_data[j];
		if (bf_data[j] > max) max = bf_data[j];
	}
	for (int j = 0; j < 2*N_SAMP; j++) {
		bf_data_char[j] = (bf_data[j] + min)/max;
	}

	// Allocate memory for the output
	float * output_f;
	output_f = (float *)calloc(N_OUTPUTS,sizeof(float));

	init_beamformer();
	update_weights(weight_filename);

	struct timespec tstart = {0,0};
	struct timespec tstop  = {0,0};
	clock_gettime(CLOCK_MONOTONIC, &tstart);

	run_beamformer(bf_data_char, output_f);

	clock_gettime(CLOCK_MONOTONIC, &tstop);
	printf("Beamformer elapsed time: %.5f seconds\n",
		((double)tstop.tv_sec + 1.0e-9*tstop.tv_nsec) -
		((double)tstart.tv_sec + 1.0e-9*tstart.tv_nsec));
	
	// Save output data to file
	FILE * output;
	output = fopen(output_filename, "w");
	fwrite(output_f, sizeof(float), N_OUTPUTS, output);
	fclose(output);

	free(output_f);
	
	return 0;
}

void printUsage() {
	printf("Usage: my_beamformer <input_filename> <weight_filename> <output_filename>\n");
}

//For makefile at the very end "-fno-exceptions -fno-rtti"
