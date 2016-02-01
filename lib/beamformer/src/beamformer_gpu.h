#ifndef BEAMFORMER_GPU_H
#define BEAMFORMER_GPU_H

// beamformer_gpu.h

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <cuComplex.h>

#define N_ELE	   38	// Number of elements/antennas in the array
#define N_BIN	   50	// Number of frequency bins
#define N_TIME	   1000	// Number of decimated time samples
#define N_BEAM     7    // Number of beams we are forming
#define N_TIME_STI 40	// Number of decimated time samples per integrated beamformer output
#define N_STI	   (N_TIME/N_TIME_STI) // Number of short time integrations
#define N_STI_BLOC 64
#define N_ELE_BLOC 64
#define N_SAMP     (N_ELE*N_BIN*N_TIME) // Number of complex samples to process
#define N_WEIGHTS  (N_ELE*N_BIN*N_BEAM) // Number of complex beamformer weights
#define N_OUTPUTS  (N_BEAM*N_STI*N_BIN) // Number of complex samples in output structure
//
#define N_TBF     (N_BEAM*N_BIN*N_TIME)
//

#define input_idx(t,f,e)     (e + f*N_ELE + (t)*N_ELE*N_BIN)
#define weight_idx(b,f,e)    (e + f*N_ELE + b*N_ELE*N_BIN)
#define sample_idx(t,b,f)    (f + b*N_BIN + (t)*N_BEAM*N_BIN)
#define output_idx(b,s,f)    (f + s*N_BIN + b*N_BIN*N_STI)
//#define mul_idx(t,b,f,e)     (e + f*N_ELE + (t)*N_ELE*N_BIN + b*N_ELE*N_BIN*N_BEAM)

/*
 * beamform
 * 
 * Takes a pointer to raw array frequency-channelized voltages, and applies a beamformer
 * using weights 
 *
 * Arguments
 * const float complex * data_in
 * 	A pointer to a 3D array, with dimensions equal to [N_TIME][N_BIN][N_ELE]
 * const double complex * weights
 * 	A pointer to a 3D array, with dimensions equal to [N_BEAM][N_BIN][N_ELE]
 * double complex * data_out
 * 	A pointer to a 3D array, with dimensions equal to [N_BEAM][N_STI][N_BIN]
 * 	(IT IS EXPECTED THAT THIS ARRAY IS INITIALIZED TO ZEROS, or else be ready for
 * 	some baffling results)
 */
/*
__global__
void beamform(const cuDoubleComplex * data_in,
              const cuDoubleComplex * weights,
	      float * data_out);
*/

__global__
void beamform(const cuFloatComplex * data_in,
              const cuFloatComplex * weights,
	          cuFloatComplex * beamformed);

__global__
void sti_reduction(const cuFloatComplex * beamformed,
	               float * data_out);


// void run_beamformer(double *data, double *weights, float *out);

#endif
