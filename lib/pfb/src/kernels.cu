#ifdef __cplusplus
extern "C" {
#include "kernels.h"
}
#endif

__global__ void map(char* dataIn,
                    char2* dataOut,
                    int channelSelect,
                    params pfbParams) 
{

    // select the channel range
    int channelMin = pfbParams.fine_channels*channelSelect;
    
    int absIdx = 2 * blockDim.y*(blockIdx.x*pfbParams.coarse_channels + (channelMin+blockIdx.y)) + 2 * threadIdx.y;  // times 2 because we are mapping a sequence of values to char2 array.
    int mapIdx = blockDim.y*(blockIdx.x*gridDim.y + blockIdx.y) + threadIdx.y;

    dataOut[mapIdx].x = dataIn[absIdx];
    dataOut[mapIdx].y = dataIn[absIdx+1];
    return;
}


/* prepare data for PFB */
__global__ void PFB_kernel(char2* pc2Data,
                      float2* pf2FFTIn,
                      float* pfPFBCoeff,
                      params pfbParams)
{
    //int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    int blkIdx = blockIdx.y * gridDim.x + blockIdx.x;
    int i = blkIdx*blockDim.x + threadIdx.x;

    int iNFFT = (gridDim.x * blockDim.x);
    int j = 0;
    int iAbsIdx = 0;
    float2 f2PFBOut = make_float2(0.0, 0.0);
    char2 c2Data = make_char2(0, 0);

    // if ( i > 10239) {
    // 	f2PFBOut.x = 1;
    // 	f2PFBOut.y = 1;
    // 	pf2FFTIn[i] = f2PFBOut;	
    // 	return;
    // }
    int coeffIdx = 0;
    int absCoeff = (blockIdx.x * blockDim.x) + threadIdx.x;
    for (j = 0; j < pfbParams.taps; ++j)
    {
        /* calculate the absolute index */
        iAbsIdx  = (j * iNFFT) + i;

     	coeffIdx = (j * iNFFT) + absCoeff;
        /* get the address of the block */
        c2Data = pc2Data[iAbsIdx];
        
        f2PFBOut.x += (float) c2Data.x * pfPFBCoeff[coeffIdx];
        f2PFBOut.y += (float) c2Data.y * pfPFBCoeff[coeffIdx];
 
    }

    pf2FFTIn[i] = f2PFBOut;

    return;
}

__global__ void CopyDataForFFT(char2 *pc2Data, float2 *pf2FFTIn)
{
    //int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int blkIdx = blockIdx.y * gridDim.x + blockIdx.x;
    int i = blkIdx*blockDim.x + threadIdx.x;
    pf2FFTIn[i].x = (float) pc2Data[i].x;
    pf2FFTIn[i].y = (float) pc2Data[i].y;

    return;
}

// prepares for the next PFB.
__global__ void saveData(char2* dataIn, char2* dataOut){
	int i = blockIdx.y*(gridDim.x*blockDim.x) + blockIdx.x*blockDim.x + threadIdx.x;

	dataOut[i] = dataIn[i];

	return;
}
