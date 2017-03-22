#include "helper.h"

// File containing helper functions for main

void printUsage(const char* progName) {
	(void) printf("Usage: %s [options] <data-file>\n", progName);
    (void) printf("    -h  --help                           ");
    (void) printf("Display this usage information\n");
    (void) printf("    -b  --nsub                           ");
    (void) printf("Number of sub-bands in the data\n");
    (void) printf("    -n  --nfft <value>                   ");
    (void) printf("Number of points in FFT\n");
    (void) printf("    -w  --window <string>                ");
    (void) printf("Filter window type, hanning, cheb-win\n");
	(void) printf("    -k  --samples <value>                ");
    (void) printf("Number of time samples processed\n");
    (void) printf("    -c  --coarse <value>                 ");
    (void) printf("Number of coarse channels in data\n");
    (void) printf("    -f  --fine <value>                   ");
    (void) printf("Number of channels selected to process\n");
    (void) printf("    -e  --elements <value>               ");
    (void) printf("Number of elements in data\n");
    (void) printf("    -d  --datatype <string>              ");
    (void) printf("Filter coefficient data type, float or int\n");
    (void) printf("    -s  --select <value>                 ");
    (void) printf("Where in channels to begin selecting fine\n");
	return;
}


int loadData(char* f, char* inputData, int size) {
	int ret = EXIT_SUCCESS;
	int file =  0;

	//int readSize = SAMPLES * DEF_NUM_CHANNELS * DEF_NUM_ELEMENTS * (2*sizeof(char));
	//inputData = (char*) malloc(readSize);
	if(NULL == inputData) {
		(void) fprintf(stderr, "ERROR: Memory allocation failed! %s.\n", strerror(errno));
		return EXIT_FAILURE;
	}

	file = open(f, O_RDONLY);
	if (file < EXIT_SUCCESS) {
		(void) fprintf(stderr, "ERROR: failed to open data file. %s\n", strerror(errno));
		return EXIT_FAILURE;
	}

	ret = read(file, inputData, size);
	if (ret < EXIT_SUCCESS) {
		(void) fprintf(stderr, "ERROR: failed to read data file. %s\n", strerror(errno));
		(void) close(file);
		return EXIT_FAILURE;
	}

	(void) close(file);
	return EXIT_SUCCESS;

}