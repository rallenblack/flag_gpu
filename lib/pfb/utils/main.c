#include "main.h"

char* g_inputData = NULL;
float2* g_outputData = NULL;

params pfbParams = DEFAULT_PFB_PARAMS;

int main(int argc, char *argv[]) {

	int ret = EXIT_SUCCESS;

	/*********************************** PARSE INPUT *****************************************/

	/* valid short and long options */
	const char* const pcOptsShort = ":hn:t:w:k:c:f:e:b:d:s:p";
	const struct option stOptsLong[] = {
		{ "help",		0, NULL,	'h' },   
		{ "nfft", 		1, NULL,	'n' },
		{ "taps",		1, NULL,	't' },
		{ "window",		1, NULL,	'w' },
		{ "samples",	1, NULL,	'k' },
		{ "coarse", 	1, NULL,	'c' },
		{ "fine",		1, NULL, 	'f' },
		{ "elements",	1, NULL,	'e' },
		{ "nsub",		1, NULL,	'b' },
		{ "datatype",	1, NULL,	'd' },
		{ "select",		1, NULL,	's' },
		{ "plot",		0, NULL,	'p' },
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

			case 'n':
				pfbParams.nfft = (int) atoi(optarg);
				break;

			case 't':
				pfbParams.taps = (int) atoi(optarg);
				break;

			case 'w':
				pfbParams.window = optarg;
				break;

			case 'k':
				pfbParams.samples = (int) atoi(optarg);
				break;

			case 'c':
				pfbParams.coarse_channels = (int) atoi(optarg);
				break;

			case 'e':
				pfbParams.elements = (int) atoi(optarg);
				break;

			case 'f':
				pfbParams.fine_channels = (int) atoi(optarg);
				break;

			case 'b':
				pfbParams.subbands =  (int) atoi(optarg);
				break;

			case 'd':
				pfbParams.dataType = optarg;
				break;

			case 's':
				pfbParams.select = (int) atoi(optarg);
				// check valid select range.
				if(pfbParams.select < 0 || pfbParams.select > 4) {
					(void) fprintf(stderr, "ERROR: Channel select range [0, 4]\n");
					errFlag++;
				}
				break;

			case 'p':
				pfbParams.plot = 0;
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
	int genFlag = 0;
	if(argc <= optind) {
		(void) fprintf(stderr, "ERROR: Missing data file.\n");
		genFlag = 1;
	}
	
	// init input data array
	int readSize = pfbParams.samples * pfbParams.coarse_channels * pfbParams.elements * (2*sizeof(char));
	g_inputData = (char*) malloc(readSize);
	memset(g_inputData, 0, readSize);

	// Determine wether to get data from a file or generate it. 
	if(!genFlag) {
		// get data filename
		char filename[256] = {0};
		(void) strncpy(filename, argv[optind], 256);
		filename[255] = '\0';
		ret = loadData(filename, g_inputData, readSize);
		if (ret == EXIT_FAILURE) {
			return EXIT_FAILURE;
		}
	} else {
		// generate data
		//generate freq array for data
		int i = 0;
		float fs = 303.0; // KHz - a default sample rate.
		int channelBandgap = 10.0;		// KHz jumps
		float* freq = (float *) malloc(pfbParams.coarse_channels*sizeof(float));
		for(i = 0; i <= pfbParams.coarse_channels; i++) {
			freq[i] = channelBandgap * i + 5.0;
		}
		genData(g_inputData, freq, fs, pfbParams.samples, pfbParams.coarse_channels, pfbParams.elements);
	}		

	/****************************** SETUP PFB ******************************/

	/* init cuda device */
	int iCudaDevice = DEF_CUDA_DEVICE;

	// create coeff and write to a file that is read in initPFB.
	pfbParams.subbands = pfbParams.elements*pfbParams.fine_channels;
	genCoeff(argc, argv, pfbParams);

	// init the device, loads coeff
	ret = initPFB(iCudaDevice, pfbParams);

	// malloc data arrays
	int outputSize = pfbParams.samples * pfbParams.fine_channels * pfbParams.elements * (2*sizeof(float)); // need to convince myself of this output data size.

	g_outputData = (float2*) malloc(outputSize);
	memset(g_outputData, 0, outputSize);


	// run the pfb function
	clock_t start, end;
	start = clock();
	ret = runPFB(g_inputData, g_outputData, pfbParams);
	end = clock();
	double timeTaken = 0;
	timeTaken = ((double) (end - start))/CLOCKS_PER_SEC;

	(void) printf("Time taken (barring Init()): %gs\n", timeTaken);

	if (ret == EXIT_FAILURE) {
		(void) fprintf(stderr, "ERROR: runPFB failed!\n");
		free(g_inputData);
		free(g_outputData);
		return EXIT_FAILURE;
	}

	// process return from pfb - write to file
	int file = 0;
	
	char outfile[256] = "output/outfile.dat\0";
	file = open(outfile,
					O_CREAT | O_TRUNC | O_WRONLY,
					S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
	if(file < EXIT_SUCCESS) {
		(void) fprintf(stderr, "ERROR: writing outfile failed\n");
		free(g_inputData);
		free(g_outputData);
		return EXIT_FAILURE;
	}

	// clean up and exit
	cleanUp();
	ret = resetDevice();

	(void) write(file, g_outputData, outputSize);
	(void) close(file);

	free(g_inputData);
	free(g_outputData);

	return EXIT_SUCCESS;
}