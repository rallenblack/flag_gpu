#ifdef __cplusplus
extern "C" {
#include "pfb.h"
}
#endif

// data ptrs
char2* g_pc2InBuf = NULL;
char2* g_pc2InBufRead = NULL;

char2* g_pc2Data_d = NULL;
char2* g_pc2DataRead_d = NULL;

float2* g_pf2FFTIn_d = NULL;
float2* g_pf2FFTOut_d = NULL;
float2* g_pf2DiscShift_d = NULL;

float *g_pfPFBCoeff = NULL;
float *g_pfPFBCoeff_d = NULL;
char* g_pcInputData_d = NULL;

// pfb params
int g_iNFFT = DEF_LEN_SPEC;
int g_iNTaps = NUM_TAPS;
int g_iNumSubBands = PFB_CHANNELS * DEF_NUM_ELEMENTS;

// process flags
int g_IsDataReadDone = FALSE;
int g_IsProcDone = FALSE;

// size vars
int g_iSizeFile = 0;
int g_iReadCount = 0;
int g_iSizeRead = DEF_SIZE_READ;
int g_iFileCoeff = 0;
char g_acFileCoeff[256] = {0};

// GPU params
dim3 g_dimBPFB(1, 1, 1);
dim3 g_dimGPFB(1, 1);
dim3 g_dimBDiscShift(1, 1, 1);
dim3 g_dimGDiscShift(1, 1, 1);
dim3 g_dimBCopy(1, 1, 1);
dim3 g_dimGCopy(1, 1);
dim3 mapGSize(1,1,1);
dim3 mapBSize(1,1,1);
dim3 saveGSize(1, 1, 1 ); // (5, 256, 1)
dim3 saveBSize(1, 1, 1); // (64, 1, 1)

cufftHandle g_stPlan = {0};
int g_iMaxThreadsPerBlock = 0;
int g_iMaxPhysThreads = 0;

int runPFB(signed char* inputData_h, float* outputData_h, params pfbParams) {

	g_IsProcDone = FALSE;
	int iRet = EXIT_SUCCESS;
	long lProcData = 0;
	long ltotData = pfbParams.fine_channels*pfbParams.elements*(pfbParams.samples + pfbParams.nfft*pfbParams.taps);
	int start = pfbParams.fine_channels*pfbParams.elements*pfbParams.nfft*pfbParams.taps;
	int countFFT = 0;
	int cpySize = pfbParams.fine_channels*pfbParams.elements*pfbParams.samples*(2*sizeof(char));

	// copy data to device
	//CUDASafeCallWithCleanUp(cudaMemcpy(g_pcInputData_d, inputData_h, g_iSizeRead, cudaMemcpyHostToDevice)); //g_iSizeRead = samples*coarse_channels*elements*(2*sizeof(char));
	CUDASafeCallWithCleanUp(cudaMemcpy(&g_pc2Data_d[start], inputData_h, cpySize, cudaMemcpyHostToDevice));

	// map - extract channel data from full data stream and load into buffer.
	//map<<<mapGSize, mapBSize>>>(g_pcInputData_d, &g_pc2Data_d[start], pfbParams.select, pfbParams);
	//CUDASafeCallWithCleanUp(cudaGetLastError());

	// Begin PFB
	g_pc2DataRead_d = g_pc2Data_d; // p_pc2Data_d contains all the data. DataRead will update with each pass through the PFB.
	int pfb_on = 1; // Enable pfb flag. Extendable.

	if(pfb_on) {
		//PFB
		PFB_kernel<<<g_dimGPFB, g_dimBPFB>>>(g_pc2DataRead_d, g_pf2FFTIn_d, g_pfPFBCoeff_d, pfbParams);
		CUDASafeCallWithCleanUp(cudaThreadSynchronize());

	} else {
		// Prepare for FFT
		CopyDataForFFT<<<g_dimGPFB, g_dimBPFB>>>(g_pc2DataRead_d, g_pf2FFTIn_d);
		CUDASafeCallWithCleanUp(cudaGetLastError());
	}

	//float2* fftOutPtr = g_pf2FFTOut_d;
	while(!g_IsProcDone) {
		//FFT
		iRet = doFFT();
                // New Code ///////////////////////////////////
                // Each run through the kernel is 1 STI window of the now, 125 (countFFT = STI window)
                Discard_Shift_kernel<<<g_dimGDiscShift, g_dimBDiscShift>>>(g_pf2FFTOut_d, g_pf2DiscShift_d);
		//CUDASafeCallWithCleanUp(cudaGetLastError());
		CUDASafeCallWithCleanUp(cudaThreadSynchronize());
               /////////////////////////////////////////////////

		if(iRet != EXIT_SUCCESS) {
			(void) fprintf(stderr, "ERROR: FFT failed\n");
			cleanUp();
			return EXIT_FAILURE;
		}
		CUDASafeCallWithCleanUp(cudaGetLastError());
		++countFFT;

		// step input and output buffers.
		g_pf2FFTIn_d += g_iNumSubBands * g_iNFFT;
		g_pf2FFTOut_d += g_iNumSubBands * g_iNFFT;
                g_pf2DiscShift_d += g_iNumSubBands * (g_iNFFT/2);

		lProcData += g_iNumSubBands * g_iNFFT;
		if(lProcData >= ltotData - NUM_TAPS*g_iNumSubBands*g_iNFFT){ // >= process 117 ffts leaving 256 time samples, > process 118 ffts leaving 224 time samples.
			g_IsProcDone = TRUE;
		}

	}
	
	// prepare next filter
	g_pc2DataRead_d += countFFT*g_iNumSubBands*g_iNFFT;
	saveData<<<saveGSize, saveBSize>>>(g_pc2DataRead_d, g_pc2Data_d);
	CUDASafeCallWithCleanUp(cudaGetLastError());

	// copy back to host.

	//wind back in/out ptrs - should put in another pointer as a process read ptr instead of updating the global ptr.
        g_pf2DiscShift_d = g_pf2DiscShift_d - countFFT*g_iNumSubBands*(g_iNFFT/2);
	g_pf2FFTOut_d = g_pf2FFTOut_d - countFFT*g_iNumSubBands*g_iNFFT;
	g_pf2FFTIn_d = g_pf2FFTIn_d -countFFT*g_iNumSubBands*g_iNFFT;

	//int outDataSize = countFFT * g_iNumSubBands * g_iNFFT;
        // Modified variable outDataSize 1/2 of g_iNFFT due to the discard of half the channels //////
	int outDataSize = countFFT * g_iNumSubBands * (g_iNFFT/2);
        
        //printf("\tData out of the DiscShift kernel: %d\n",outDataSize);
        //printf("\tcountFFT: %d\n",countFFT);
        //printf("\tNFFT: %d\n",g_iNFFT);
        //printf("\tcufftComplex: %d\n",sizeof(cufftComplex));
        //printf("\tfloat2: %d\n",sizeof(float2));

	//CUDASafeCallWithCleanUp(cudaMemcpy(outputData_h, g_pf2FFTOut_d, outDataSize*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
	CUDASafeCallWithCleanUp(cudaMemcpy(outputData_h, g_pf2DiscShift_d, outDataSize*sizeof(cufftComplex), cudaMemcpyDeviceToHost));
        ////////////////////////////////////////////////////////////////////////////////////////////

	return iRet;

}

void flushBuffer(params pfbParams) {

	int start = pfbParams.fine_channels*pfbParams.elements*pfbParams.nfft*pfbParams.taps;
	CUDASafeCallWithCleanUp(cudaMemset((void *)   g_pc2Data_d, 0, start*2*sizeof(char)));
	return;
}

// return true or false upon successful setup.
int initPFB(int iCudaDevice, params pfbParams){

	int iRet = EXIT_SUCCESS;

	// set pfb params from input parameters.
	pfbParams.subbands = pfbParams.elements*pfbParams.fine_channels;

	g_iNFFT = pfbParams.nfft;
	g_iNTaps = pfbParams.taps;
	g_iNumSubBands = pfbParams.subbands; // equal to elements*fine_channels. (The fine channels are the channels processed.)

	g_iSizeRead = pfbParams.samples*pfbParams.coarse_channels*pfbParams.elements*(2*sizeof(char));

	char* coeffLoc = pfbParams.coeffPath;

	int iDevCount = 0;
	cudaDeviceProp stDevProp = {0};
	cufftResult iCUFFTRet = CUFFT_SUCCESS;

	int i = 0;

	//Register signal handlers?

	/********************************************/
	/* Look for eligable Cuda Device and select */
	/********************************************/
	(void) fprintf(stdout, "Querying CUDA devices.\n");

	(void) cudaGetDeviceCount(&iDevCount);
	if (0 == iDevCount) {
		(void) fprintf(stderr, "ERROR: No CUDA-capable device found!\n");
		return EXIT_FAILURE;
	}
	// Look for requested device (if applicable)
	if (iCudaDevice >= iDevCount) {
		(void) fprintf(stderr,
					   "ERROR: Requested device %d no found in present %d device list.\n",
					   iCudaDevice,
					   iDevCount);
		return EXIT_FAILURE;
	}
	// Query devices and setup selected device.
	for(i = 0; i < iDevCount; i++) {
		CUDASafeCallWithCleanUp(cudaGetDeviceProperties(&stDevProp, i));
		printf("\tDevice %d: %s, Compute Capability %d.%d, %d physical threads %s\n",
				i,
				stDevProp.name, stDevProp.major, stDevProp.minor,
				stDevProp.multiProcessorCount * stDevProp.maxThreadsPerMultiProcessor,
				(iCudaDevice == i) ? "<<SELECTED>>" : "");
	}
	CUDASafeCallWithCleanUp(cudaSetDevice(iCudaDevice));

	// Setup block and thread paramters
	CUDASafeCallWithCleanUp(cudaGetDeviceProperties(&stDevProp, 0));
	g_iMaxThreadsPerBlock = stDevProp.maxThreadsPerBlock;
	g_iMaxPhysThreads = stDevProp.multiProcessorCount * stDevProp.maxThreadsPerMultiProcessor;

	// Check if valid operation lengths. i.e. The input buffer is long enough (should this be done here or elsewhere?)

	// Set malloc size - lTotCUDAMalloc is used only to calculate the total amount of memory not used for the allocation.
	size_t cudaMem_total, cudaMem_available;
	size_t lTotCUDAMalloc = 0;
	cudaMemGetInfo(&cudaMem_available, &cudaMem_total);
	lTotCUDAMalloc += g_iSizeRead; // size   data
	lTotCUDAMalloc += (g_iNumSubBands * pfbParams.samples * sizeof(float(2))); // size of FFT input array This should be different since our data is unsigned char?
	lTotCUDAMalloc += (g_iNumSubBands * pfbParams.samples * sizeof(float(2))); // size of FFT output array
	lTotCUDAMalloc += (g_iNumSubBands * (pfbParams.samples/2) * sizeof(float(2))); // size of Discard & shift output array
	lTotCUDAMalloc += (g_iNumSubBands * g_iNFFT * sizeof(float)); 	// size of PFB Coefficients
	// Check CUDA device can handle the memory request
	if(lTotCUDAMalloc > stDevProp.totalGlobalMem) {
		(void) fprintf(stderr,
						"ERROR: Total memory requested on GPU is %g MB of %g possible MB (Total Global Memory: %g MB).\n"
						"\t**** Memory breakdown *****\n"
						"\tInput data buffer:\t%g MB\n"
						"\tFFT in array:\t%g MB\n"
						"\tFFT out array:\t%g MB\n"
						"\tDiscard and shift array:\t%g MB\n"
						"\tPFB Coefficients: %f KB\n",
						((float) lTotCUDAMalloc) / (1024*1024),
						((float) cudaMem_available) / (1024*1024), //stDevProp.totalGlobalMem
						((float) cudaMem_total) / (1024*1024),
						((float) g_iSizeRead) / (1024 * 1024),
						((float) g_iNumSubBands * pfbParams.samples * sizeof(float2)) / (1024 * 1024),
						((float) g_iNumSubBands * pfbParams.samples * sizeof(float2)) / (1024 * 1024),
						((float) g_iNumSubBands * (pfbParams.samples/2) * sizeof(float2)) / (1024 * 1024),
						((float) g_iNumSubBands * g_iNFFT * sizeof(float)));
		return EXIT_FAILURE;
	}
	
	// print memory usage report.
	(void) fprintf(stdout,
					"INFO: Total memory requested on GPU is %g MB of %g possible MB (Total Global Memory: %g MB).\n"
					"\t**** Memory breakdown ****\n"
					"\tInput data buffer:\t%g MB\n"
					"\tFFT in array:\t%g MB\n"
					"\tFFT out array:\t%g MB\n" // Half of input array due to discard of half of the 64 hcannels.
					"\tDiscard and shift array:\t%g MB\n"
					"\tPFB Coefficients: %f KB\n",
					((float) lTotCUDAMalloc) / (1024*1024),
					((float) cudaMem_available) / (1024*1024), //stDevProp.totalGlobalMem
					((float) cudaMem_total) / (1024*1024),
					((float) g_iSizeRead) / (1024 * 1024),
					((float) g_iNumSubBands * pfbParams.samples * sizeof(float2)) / (1024 * 1024),
					((float) g_iNumSubBands * pfbParams.samples * sizeof(float2)) / (1024 * 1024),
					((float) g_iNumSubBands * (pfbParams.samples/2) * sizeof(float2)) / (1024 * 1024),
					((float) g_iNumSubBands * g_iNFFT * sizeof(float)));

	/*************************/
	/* Load PFB coefficients */
	/*************************/
	(void) fprintf(stdout, "\nSetting up PFB filter coefficients...\n");
	int sizePFB = g_iNumSubBands * g_iNTaps * g_iNFFT * sizeof(float);

	// Allocate memory for PFB coefficients to be read in
	g_pfPFBCoeff = (float *) malloc(sizePFB); // allocate the memory needed for the size of one pfb pass through
	if(NULL == g_pfPFBCoeff) {
		(void) fprintf(stderr, "ERROR: Memory allocation for the PFB coefficients failed. %s\n",
								strerror(errno));
		return EXIT_FAILURE;
	}
	
	// Read filter coefficients from file
	(void) fprintf(stdout, "\tReading in coefficients...\n");
	(void) sprintf(g_acFileCoeff,
				   "%s%s_%s_%d_%d_%d%s",
				   coeffLoc,
				   FILE_COEFF_PREFIX,
				   FILE_COEFF_DATATYPE,
				   g_iNTaps,
				   g_iNFFT,
				   g_iNumSubBands,
				   FILE_COEFF_SUFFIX);

	g_iFileCoeff = open(g_acFileCoeff, O_RDONLY);
	if(g_iFileCoeff < EXIT_SUCCESS) {
		(void) fprintf(stderr, "ERROR: Failed to open coefficient file %s. %s\n",
					  			g_acFileCoeff,
					  			strerror(errno));
		return EXIT_FAILURE;
	}

	iRet = read(g_iFileCoeff, g_pfPFBCoeff, sizePFB);
	if(iRet != sizePFB) {
		(void) fprintf(stderr, "ERROR: Failed reading filter coefficients. %s\n", strerror(errno));
		return EXIT_FAILURE;
	}
	
	(void) close(g_iFileCoeff);

	/********************************************/
	/* Allocate memory and setup on CUDA device */
	/********************************************/
	(void) fprintf(stdout, "\nSetting up CUDA device.\n");

	//malloc map array and copy data to device
	(void) fprintf(stdout, "\tAllocating memory for MAP...\n");
	// creates a size that is paddedd in the front to store the filter state. Worth one 256 (nfft*taps) time sample amount of data
	int sizeMap = pfbParams.samples * pfbParams.fine_channels * pfbParams.elements * (2*sizeof(char)) + pfbParams.fine_channels*pfbParams.elements*pfbParams.nfft*pfbParams.taps * (2*sizeof(char));
	CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pcInputData_d, g_iSizeRead));
	CUDASafeCallWithCleanUp(cudaMemset((void *)   g_pcInputData_d, 0, g_iSizeRead));
	CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pc2Data_d, sizeMap));
	CUDASafeCallWithCleanUp(cudaMemset((void *)   g_pc2Data_d, 0, sizeMap));

	// allocate memory for pfb coefficients on GPU
	(void) fprintf(stdout, "\tAllocating memory for PFB...\n");
	CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pfPFBCoeff_d, sizePFB));

	// copy coeff to device
	(void) fprintf(stdout, "\tCopying filter coefficients...\n");
	CUDASafeCallWithCleanUp(cudaMemcpy(g_pfPFBCoeff_d, g_pfPFBCoeff, sizePFB, cudaMemcpyHostToDevice));

	// allocate memory for FFT in and out arrays
	(void) fprintf(stdout, "\tAllocate memory for FFT arrays...\n");
	//int sizeDataBlock_in = g_iNumSubBands * g_iNFFT * sizeof(float2);
	int sizeDataBlock_in = pfbParams.samples*g_iNumSubBands * sizeof(float2);
	int sizeTotalDataBlock_out = pfbParams.samples*g_iNumSubBands * sizeof(float2); // output fft array same size as output data for convinence the full size is not used. In the pfb function the output data will be the fft counter times block amount in the fft.
        // New variables ///////////////////////////////////
        int g_iNwindows = 125; // with 8000 time samples, but with 4032 samples, Nwindows = 63
        int sizeTotalDataBlock_disc = (g_iNFFT/2)*g_iNwindows*g_iNumSubBands * sizeof(float2); // Not sure about the value of this yet.
        //////////////////////////////////////////////////
	CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf2FFTIn_d, sizeDataBlock_in));
	CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf2FFTOut_d, sizeTotalDataBlock_out)); // goal will be to update the output ptr each time it fires.
        // New Code ////////////////////////////////////////////////////////
        CUDASafeCallWithCleanUp(cudaMalloc((void **) &g_pf2DiscShift_d, sizeTotalDataBlock_disc)); // Discard and shift array
        ////////////////////////////////////////////////////////////////////
	CUDASafeCallWithCleanUp(cudaMemset((void *) g_pf2FFTIn_d, 0, sizeDataBlock_in));
	CUDASafeCallWithCleanUp(cudaMemset((void *) g_pf2FFTOut_d, 0, sizeTotalDataBlock_out));
        // New Code ////////////////////////////////////////////////////////////
        CUDASafeCallWithCleanUp(cudaMemset((void *) g_pf2DiscShift_d, 0, sizeTotalDataBlock_disc));
        ////////////////////////////////////////////////////////////////////////

	// set kernel parameters
	(void) fprintf(stdout, "\tSetting kernel parameters...\n");
	if(g_iNFFT < g_iMaxThreadsPerBlock) {
		g_dimBPFB.x  = g_iNFFT;
		//g_dimBDiscShift.x  = g_iNFFT;
		g_dimBDiscShift.x  = g_iNumSubBands;
		g_dimBCopy.x = g_iNFFT;
	} else {
		g_dimBPFB.x  = g_iMaxThreadsPerBlock;
		g_dimBDiscShift.x  = g_iMaxThreadsPerBlock;
		g_dimBCopy.x = g_iMaxThreadsPerBlock;
	}
	g_dimGPFB.x  = (g_iNumSubBands * g_iNFFT) / g_dimBPFB.x;
	//g_dimGDiscShift.x  = (g_iNumSubBands * g_iNFFT) / g_dimBDiscShift.x;
	g_dimGDiscShift.x  = g_iNFFT;
	g_dimGCopy.x = (g_iNumSubBands * g_iNFFT) / g_dimBCopy.x;

	g_dimGPFB.y = 125; // with 8000 time samples, but with 4032 samples, g_dimBPFB.y = 63
	// g_dimGDiscShift.y = 125; // with 8000 time samples, but with 4032 samples, g_dimBPFB.y = 63
	g_dimGCopy.y = 125; // same as g_dimGPFB.y

        //g_dimGDiscShift.z = 2; // Chunks of channels to recover

        // g_dimBPFB.y = 63; // No. of windows given 64 point FFTs and 4000 time samples per block. Since 4000/64=62.5, I need to see whether 32 samples should be discarded.
        // g_dimGPFB.y = 63; // No. of windows given 64 point FFTs and 4000 time samples per block

	// map kernel params	
	mapGSize.x = pfbParams.samples;
	mapGSize.y = pfbParams.fine_channels;
	mapGSize.z = 1;

	mapBSize.x = 1;
	mapBSize.y = pfbParams.elements;
	mapBSize.z = 1;

	// copy kernel params
	saveGSize.x = pfbParams.fine_channels;
	saveGSize.y = pfbParams.nfft*pfbParams.taps;
	saveGSize.z = 1;

	saveBSize.x = pfbParams.elements;
	saveBSize.y = 1;
	saveBSize.z = 1;

	(void) fprintf(stdout, "\t\tPFB Kernel Parmaters are:\n\t\tgridDim(%d,%d,%d) blockDim(%d,%d,%d)\n\n",
							g_dimGPFB.x, g_dimGPFB.y, g_dimGPFB.z,
							g_dimBPFB.x, g_dimBPFB.y, g_dimBPFB.z);

	(void) fprintf(stdout, "\t\tDiscard/Shift Kernel Parmaters are:\n\t\tgridDim(%d,%d,%d) blockDim(%d,%d,%d)\n\n",
							g_dimGDiscShift.x, g_dimGDiscShift.y, g_dimGDiscShift.z,
							g_dimBDiscShift.x, g_dimBDiscShift.y, g_dimBDiscShift.z);

	(void) fprintf(stdout, "\t\tMAP Kernel Parmaters are:\n\t\tgridDim(%d,%d,%d) blockDim(%d,%d,%d)\n\n",
							mapGSize.x, mapGSize.y, mapGSize.z,
							mapBSize.x, mapBSize.y, mapBSize.z);

	(void) fprintf(stdout, "\t\tSave Kernel Parmaters are:\n\t\tgridDim(%d,%d,%d) blockDim(%d,%d,%d)\n",
							saveGSize.x, saveGSize.y, saveGSize.z,
							saveBSize.x, saveBSize.y, saveBSize.z);

	// create a CUFFT plan

	(void) fprintf(stdout, "\tCreating cuFFT plan...\n");
	iCUFFTRet = cufftPlanMany(&g_stPlan,
							  FFTPLAN_RANK,
							  &g_iNFFT,
							  &g_iNFFT,
							  FFTPLAN_ISTRIDE,
							  FFTPLAN_IDIST,
							  &g_iNFFT,
							  FFTPLAN_OSTRIDE,
							  FFTPLAN_ODIST,
							  CUFFT_C2C,
							  FFTPLAN_BATCH);
	if(iCUFFTRet != CUFFT_SUCCESS) {
		(void) fprintf(stderr, "ERROR: Plan creation failed!\n");
		return EXIT_FAILURE;
	}

	fprintf(stdout, "\nDevice for PFB successfully initialized!\n");
	return EXIT_SUCCESS;

}

int resetDevice() {
	cudaError_t cuErr = cudaDeviceReset();
	if (cuErr != cudaSuccess) {
		fprintf(stderr, "Device Reset Failed.\n");

		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

/* do fft on pfb data */
int doFFT()
{
    cufftResult iCUFFTRet = CUFFT_SUCCESS;

    /* execute plan */
    iCUFFTRet = cufftExecC2C(g_stPlan,
                             (cufftComplex*) g_pf2FFTIn_d,
                             (cufftComplex*) g_pf2FFTOut_d,
                             CUFFT_FORWARD);
    if (iCUFFTRet != CUFFT_SUCCESS)
    {
        (void) fprintf(stderr, "ERROR! FFT failed!\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

void __CUDASafeCallWithCleanUp(cudaError_t iRet,
                               const char* pcFile,
                               const int iLine,
                               void (*pcleanUp)(void))
{
    if (iRet != cudaSuccess)
    {
        (void) fprintf(stderr,
                       "ERROR: File <%s>, Line %d: %s\n",
                       pcFile,
                       iLine,
                       cudaGetErrorString(iRet));
        /* free resources */
        (*pcleanUp)();
        exit(EXIT_FAILURE);
    }

    return;
}

void cleanUp() {
/* free resources */
    if (g_pc2InBuf != NULL) {
        free(g_pc2InBuf);
        g_pc2InBuf = NULL;
    }
    if (g_pc2Data_d != NULL) {
        (void) cudaFree(g_pc2Data_d);
        g_pc2Data_d = NULL;
    }
    if (g_pf2FFTIn_d != NULL) {
        (void) cudaFree(g_pf2FFTIn_d);
        g_pf2FFTIn_d = NULL;
    }
    if (g_pf2FFTOut_d != NULL) {
        (void) cudaFree(g_pf2FFTOut_d);
        g_pf2FFTOut_d = NULL;
    }
    if (g_pf2DiscShift_d != NULL) {
        (void) cudaFree(g_pf2DiscShift_d);
        g_pf2DiscShift_d = NULL;
    }

    free(g_pfPFBCoeff);
    (void) cudaFree(g_pfPFBCoeff_d);

    /* destroy plan */
    /* TODO: check for plan */
    (void) cufftDestroy(g_stPlan);

    return;
}















