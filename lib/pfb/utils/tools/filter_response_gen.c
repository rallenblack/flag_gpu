#include <stdio.h>

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>
#include <float.h>
#include <getopt.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

#define LEN_GENSTRING 256
#define SCALE_FACTOR  127
#define F_S		      256.0 // MHz
#define N			  256  // Time samples
#define MAX_FREQ      256
#define CHANNELS	  25    // Freq Channels
#define NUM_EL		  64    // Antenna Elements

void printUsage(const char* progName) {
	(void) printf("Usage: %s [options] <data-file>\n", progName);
    (void) printf("    -h  --help                           ");
    (void) printf("Display this usage information\n");
    (void) printf("    -s  --samples <value>                ");
    (void) printf("Number of samples in the data\n");
    (void) printf("    -f  --fs <value>                     ");
    (void) printf("Max freq\n");
    (void) printf("    -w  --band <value>                   ");
    (void) printf("Sample rate\n");
    (void) printf("    -c  --channels <value>               ");
    (void) printf("Number of channels in data\n");
    (void) printf("    -e  --elemnets <value>               ");
    (void) printf("Number of elements in data\n");
	return;
}

int main(int argc, char *argv[]) {

	int samples = N;
	int fs = F_S;
	int coarseChannels = CHANNELS;
	int elements = NUM_EL;
	int genFreq = MAX_FREQ;

	/* valid short and long options */
	const char* const pcOptsShort = ":hs:f:w:c:e:";
	const struct option stOptsLong[] = {
		{ "help",		0, NULL,	'h' },
		{ "samples",	1, NULL,	's' },
		{ "fs",			1, NULL,	'f' },
		{ "band",		1, NULL,	'w' },
		{ "channels",	1, NULL,	'c' },
		{ "elements",	1, NULL,	'e' },
		{ NULL,			0, NULL, 	0	}
	};

	const char* progName = argv[0];

	int errFlag = 0;

	/* parse input */
	int opt = 0; //
	int prevInd = 0; // used to track optind to manual check missing arguments.
	do {
		/* 
			Getopt will load the next option if the argument is missing, getopt's ':' error check
			really only works on the last option. This assumes that no argument has a '-' in it.
		*/
		prevInd = optind;
		opt = getopt_long(argc, argv, pcOptsShort, stOptsLong, NULL);

		if(optind == prevInd + 2 && (*optarg == '-' || *optarg == '.')) { // assumes arguments cannot start with '-' or '.'. Also, if optarg is null this causes a seg fault and the first logical comparisson catches the null case. The parans for the or helps not cause the fault.
			optopt = opt; // update getopt's optopt variable to contain the violating variable. 
			opt = ':'; // trigger the error character.
			--optind; // decrement optind since it was incremented incorrectly.
		}

		switch(opt)
		{
			case 'h':
				printUsage(progName);
				return EXIT_SUCCESS;

			case 's':
				samples = (int) atoi(optarg);
				break;

			case 'c':
				coarseChannels = (int) atoi(optarg);
				break;

			case 'e':
				elements = (int) atoi(optarg);
				break;

			case 'f':
				fs = (int) atoi(optarg);
				break;

			case 'w':
				genFreq = (int) atoi(optarg);
				break;

			case ':':
				(void) fprintf(stderr, "-%c option requires a parameter.\n", optopt);
				errFlag++;
				break;

			case '?':
				(void) fprintf(stderr, "Unrecognized option -%c.\n", optopt);
				errFlag++;
				break;

			case -1: /* done with options */
				break;

			default: /* unexpected */
				assert(0);
		}
	} while (opt != -1);

	if(errFlag) {
		printUsage(progName);
		return EXIT_FAILURE;
	}

	// no data file presented
	if(argc <= optind) {
		(void) fprintf(stderr, "ERROR: Missing output data filename.\n");
		return EXIT_FAILURE;
	}

	int iFile = 0;
	char acDataFilename[LEN_GENSTRING] = {0};
	(void) strncpy(acDataFilename, argv[optind], LEN_GENSTRING);
	acDataFilename[LEN_GENSTRING-1] = '\0'; //NUll terminator at end of filename string.

	iFile = open(acDataFilename,
					O_CREAT | O_TRUNC | O_WRONLY,
					S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
	if(EXIT_FAILURE == iFile) {
		(void) fprintf(stderr, "ERROR: Failed to open output file. %s\n", strerror(errno));
		return EXIT_FAILURE;
	}

	int i = 0;
	//const int genFreq = 256;
    //Generate freq array.
    float* freq = NULL;
    freq = (float*) malloc(genFreq*sizeof(float));
    //float freq[genFreq] = {};
    for(i = 1; i <= genFreq; i++) {
        freq[i-1] = i*1.0;
        //fprintf(stdout, "freq: %f\n", freq[i-1]);
    }
    fprintf(stdout, "i: %d\n", i);
    int f = 0;
	int n = 0;
	int c = 0;
	int e = 0;
 
	signed char cDataRe = 0;
	signed char cDataIm = 0;
	signed char* toWrite = (signed char *) malloc(samples*coarseChannels*elements*(2*sizeof(signed char)));

	fprintf(stdout,
			"INFO: Generating samples...\n"
			"\tSamples:\t %d\n"
			"\tSample rate:\t %d\n"
			"\tMax Range:\t %d\n"
			"\tChannels:\t %d\n"
			"\tElements:\t %d\n",
			samples, fs, genFreq, coarseChannels, elements);

	for(f = 0; f < genFreq; f++) {
		for(n = 0; n < samples; n++) {

				cDataRe = SCALE_FACTOR * (0.1 * cos(2*M_PI * freq[f] * n / fs));
				cDataIm = SCALE_FACTOR * (0.1 * sin(2*M_PI * freq[f] * n / fs));

			for(c = 0; c < coarseChannels; c++) {
				for(e = 0; e < 2*elements; e++) {
					
					int idx = e + c * (2 * elements) + n * (coarseChannels * 2*elements);
					if( !(e%2) ) {
					//create interleaved samples for real and Im 
					toWrite[idx] = cDataRe;
					} else {
					toWrite[idx] = cDataIm;
					}
				}
			}
		}
		(void) write(iFile, toWrite, samples*coarseChannels*elements*(2*sizeof(signed char)));
	}

	(void) close(iFile);

	return EXIT_SUCCESS;
}