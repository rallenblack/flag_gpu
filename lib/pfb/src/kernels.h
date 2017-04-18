#ifndef __KERNELS_H__
#define __KERNELS_H__

#include <cuda.h>
#include <cufft.h>

// stuct of parameters for PFB. Values indicate default values.
#define DEFAULT_PFB_PARAMS {4000, 32, 8, 25, 5, 64, 320, 0, (char*)"hanning\0", (char*)"float\0", (char*)"\0",  1};
// plot 1 mean to hide the plot of the filter before continuing.
typedef struct {
	int samples;
	int nfft;
	int taps;
	int coarse_channels;
	int fine_channels;
	int elements;
	int subbands;
	int select;
	char* window;
	char* dataType;
	char* coeffPath;
	int plot;
} params;


__global__ void PFB_kernel(char2* pc2Data, float2* pf2FFTIn, float* pfPFBCoeff, params pfbParams);
__global__ void map(char* dataIn, char2* dataOut, int channelSelect, params pfbParams);
__global__ void CopyDataForFFT(char2* pc2Data, float2* pf2FFTIn);
__global__ void saveData(char2* dataIn, char2* dataOut);

#endif
