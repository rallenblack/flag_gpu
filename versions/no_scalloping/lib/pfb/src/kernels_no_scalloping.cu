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
    int blkIdx = blockIdx.y * gridDim.x + blockIdx.x;
    int i = blkIdx*blockDim.x + threadIdx.x;

    int absCoeff = (blockIdx.x * blockDim.x) + threadIdx.x;

    int iNFFT = (gridDim.x * blockDim.x);
    int j = 0;
    int iAbsIdx = 0;
    int coeffIdx = 0;

    float2 f2PFBOut = make_float2(0.0, 0.0);
    char2 c2Data = make_char2(0, 0);

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

// Discard channels and perform FFT shift (part of scalloping solution)
//__global__ void Discard_Shift_kernel(float2* pf2FFTOut, float2* pf2DiscShift)
//{
//    int pt = threadIdx.x; // N-point FFT index (0:63)
//    int sb = blockIdx.x;  // Number of elements x coarse channels (time series) index (0:319)
//    //int st = blockIdx.y;  // Windows index (4000/32 = 125 windows) (0:124)
//    int Nfft = blockDim.x; // N-point FFT  (64)
//    //int Nsubbands = gridDim.x; // Nele*NfineChannels (64*5=320)
////    int Nchunks = 2;
////    int i = 0;
//
//    int recover_idx = 0;
//    int fftshift_idx = 0;
//
//    // // Both pre-processor macros are defined in kernels.h //////////////////////////////
//    // pf2DiscShift[fftshift_idx(pt,i,sb,st)].x = pf2FFTOut[recover_idx(pt,i,sb,st)].x;
//    // pf2DiscShift[fftshift_idx(pt,i,sb,st)].y = pf2FFTOut[recover_idx(pt,i,sb,st)].y;
//    // ////////////////////////////////////////////////////////////////////////////////////
//
////    for (i = 0; i < Nchunks; i++)
////    {
////	if (pt < (Nfft/4)) // pt indexing: 0:15 with Nfft/4 = 16
////	{
////    	    // recover_idx = (pt + (48*i)) + Nfft*sb + Nfft*Nsubbands*st;
////    	    // fftshift_idx = (pt + (16*(1-i))) + (Nfft/2)*sb + (Nfft/2)*Nsubbands*st;
////
////    	    recover_idx = (pt + (48*i)) + Nfft*sb;
////    	    fftshift_idx = (pt + (16*(1-i))) + (Nfft/2)*sb;
////
////    	    pf2DiscShift[fftshift_idx].x = pf2FFTOut[recover_idx].x;
////    	    pf2DiscShift[fftshift_idx].y = pf2FFTOut[recover_idx].y;
////        }
////    }
//
//    if (pt < (Nfft/4))
//    {
//        recover_idx = pt + Nfft*sb;
//        fftshift_idx = (pt + 16) + (Nfft/2)*sb;
//
//        pf2DiscShift[fftshift_idx].x = pf2FFTOut[recover_idx].x;
//        pf2DiscShift[fftshift_idx].y = pf2FFTOut[recover_idx].y;
//    }
//
//    if (pt >= (Nfft*(3/4)))
//    {
//        recover_idx = pt + Nfft*sb;
//        fftshift_idx = (pt - 48) + (Nfft/2)*sb;
//
//       pf2DiscShift[fftshift_idx].x = pf2FFTOut[recover_idx].x;
//        pf2DiscShift[fftshift_idx].y = pf2FFTOut[recover_idx].y;
//    }
//
//    return;
//}

// Discard channels and perform FFT shift (part of scalloping solution and altered dimensions (subbands then nfft))
__global__ void Discard_Shift_kernel(float2* pf2FFTOut, float2* pf2DiscShift)
{
    int pt = blockIdx.x; // N-point FFT index (0:63)
    int sb = threadIdx.x;  // Number of elements x coarse channels (time series) index (0:319)
    //int st = blockIdx.y;  // Windows index (4000/32 = 125 windows) (0:124)
    int Nfft = gridDim.x; // N-point FFT  (64)
    int Nsubbands = blockDim.x; // Nele*NfineChannels (64*5=320)
//    int Nchunks = 2;
//    int i = 0;

    int recover_idx = 0;
    int fftshift_idx = 0;

//    for (i = 0; i < Nchunks; i++)
//    {
//      if (pt < (Nfft/4)) // pt indexing: 0:15 with Nfft/4 = 16
//      {
//          recover_idx = sb + Nsubbands*(pt + (48*i));
//          fftshift_idx = sb + Nsubbands*(pt + (16*(1-i)));
//
//          pf2DiscShift[fftshift_idx].x = pf2FFTOut[recover_idx].x;
//          pf2DiscShift[fftshift_idx].y = pf2FFTOut[recover_idx].y;
//        }
//    }

    if (pt < (Nfft/4)) // Number of FFT points less that 16 (0:15)
    {
        recover_idx = sb + Nsubbands*pt; // Recover FFT points 0:15
        fftshift_idx = sb + Nsubbands*(pt + 16); // Place the recovered points in 16:31 of this array
        //fftshift_idx = sb + Nsubbands*pt; // Place the recovered points in 0:15 of this array (No fft shift)

        pf2DiscShift[fftshift_idx].x = pf2FFTOut[recover_idx].x;
        pf2DiscShift[fftshift_idx].y = pf2FFTOut[recover_idx].y;
    }

    if (pt >= (Nfft*(3/4))) // Number of FFT points greater that 48 (48:63)
    {
        recover_idx = sb + Nsubbands*pt; // Recover FFT points 48:63
        fftshift_idx = sb + Nsubbands*(pt - 48); // Place the recovered points in 0:15 of this array
        //fftshift_idx = sb + Nsubbands*(pt - 32); // Place the recovered points in 16:31 of this array (No fft shift)

        pf2DiscShift[fftshift_idx].x = pf2FFTOut[recover_idx].x;
        pf2DiscShift[fftshift_idx].y = pf2FFTOut[recover_idx].y;
    }

    return;
}

// When PFB disabled just perform FFT.
__global__ void CopyDataForFFT(char2 *pc2Data, float2 *pf2FFTIn)
{
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
