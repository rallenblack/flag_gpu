#ifndef __PFB_H__
#define __PFB_H__

#include <stdio.h>
#include <stdlib.h>

#include <sys/types.h>  /* for open()  */
#include <sys/stat.h>	/* for open()  */
#include <fcntl.h>		/* for open()  */
#include <errno.h>		/* for errno   */
#include <unistd.h>     /* for read, close*/

//#include <python2.7/Python.h> /* for executing coeff gen file */

#include "kernels.h"

#define FALSE 					0
#define TRUE  					1
#define DEBUG					1

#define DEF_CUDA_DEVICE			0

#define DEF_SIZE_READ			262144	// data block size. should this be set dynamically once I get the data?
#define DEF_LEN_SPEC			32	// Transform size
#define NUM_TAPS				8	// PFB Decimation factor
#define DEF_NUM_CHANNELS		25  // System spec for total number of channels
#define PFB_CHANNELS			5	// Number of coarse channels through PFB
#define DEF_NUM_ELEMENTS		64  // System spec for number of elements
#define SAMPLES					4000// Time samples.

#define PFB_OUTPUT_BLOCK_SIZE	(SAMPLES+3*DEF_LEN_SPEC)*PFB_CHANNELS*ELEMNTS*2 // (3*DEF_LEN_SPEC is to add more samples on the end to make it look like 128 pfb windows had been processed for the pfb correlator)

// FFT Plan configuration
#define FFTPLAN_RANK 			1				 // dimension of the transform
#define FFTPLAN_ISTRIDE			(g_iNumSubBands) // The distance between two successive input time elements. - (polarization*numsubbands).
#define FFTPLAN_OSTRIDE			(g_iNumSubBands) // Similar to ostride to maintain data structure
#define FFTPLAN_IDIST			1				 // The distance between the first elements of two consecutive batches in the input data. Each FFT operation is a 'batch'. Each subband is a time series and we need a FFT for each subband. Since we have interleaved samples the distance between consecutive batches is 1 sample.
#define FFTPLAN_ODIST			1				 // Simailar to odist to maintian data structure
#define FFTPLAN_BATCH			(g_iNumSubBands) // The total number of FFTs to perform per call to DoFFT().

// coeff file configuration
#define FILE_COEFF_PREFIX		"coeff"
#define FILE_COEFF_DATATYPE		"float"
#define FILE_COEFF_SUFFIX		".dat"

#define USEC2SEC            1e-6

typedef unsigned char BYTE;

#define CUDASafeCallWithCleanUp(iRet) __CUDASafeCallWithCleanUp(iRet, __FILE__, __LINE__, &cleanUp)
void __CUDASafeCallWithCleanUp(cudaError_t iRet, const char* pcFile, const int iLine, void (*pcleanUp)(void));

//void genCoeff(char* procName, params pfbParams);
int initPFB(int iCudaDevice, params pfbParams);
int runPFB(signed char* inputData_h, float* outputData_h, params pfbParams);
int doFFT();
int resetDevice(void);
void cleanUp(void);

#endif
