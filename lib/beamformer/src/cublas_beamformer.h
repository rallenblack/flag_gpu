#ifndef CUBLAS_BEAMFORMER
#define CUBLAS_BEAMFORMER

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>

#define BN_ELE	   38	// Number of elements/antennas in the array
#define BN_BIN	   25	// Number of frequency bins
#define BN_TIME	   4000	//40 // Number of decimated time samples
#define BN_BEAM     14   // Number of beams we are forming
#define BN_POL     4
#define BN_BEAM1    (BN_BEAM/2)   // Number of beams we are forming
#define BN_TIME_STI 40	//40 // Number of decimated time samples per integrated beamformer output
#define BN_STI	   (BN_TIME/BN_TIME_STI) // Number of short time integrations
#define BN_STI_BLOC 64
#define BN_ELE_BLOC 64
#define BN_SAMP     (BN_ELE_BLOC*BN_BIN*BN_TIME) // Number of complex samples to process
#define BN_WEIGHTS  (BN_ELE_BLOC*BN_BIN*BN_BEAM) // Number of complex beamformer weights
#define BN_OUTPUTS  (BN_BEAM*BN_STI*BN_BIN) // Number of complex samples in output structure
//
#define BN_TBF     (BN_BEAM*BN_BIN*BN_TIME)
//

#define input_idx(t,f,e)     ((e) + (f)*BN_ELE_BLOC + (t)*BN_ELE_BLOC*BN_BIN)
#define weight_idx(b,f,e)    ((e) + (f)*BN_ELE_BLOC + (b)*BN_ELE_BLOC*BN_BIN)
#define sample_idx(t,b,f)    ((b) + (t)*BN_BEAM + (f)*BN_BEAM*BN_TIME)
#define output_idx(p,b,s,f)    ((b) + (p)*BN_BEAM1 + (f)*BN_BEAM1*BN_POL + (s)*BN_BEAM1*BN_BIN*BN_POL)

#ifdef __cplusplus
extern "C" {
#endif
void update_weights(char * filename);
void init_beamformer();
void run_beamformer(signed char * data_in, float * data_out);
#ifdef __cplusplus
}
#endif

#endif
