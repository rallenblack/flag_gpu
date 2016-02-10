#ifndef TOTAL_POWER_H
#define TOTAL_POWER_H

#include <stdlib.h>
#include <stdio.h>

#define NF 5 		// Number of Fengines
#define NI 8 		// Number of inputs per Fengine
#define NA (NF*NI) 	// Number of total antennas
#define NC 50 		// Number of frequency channels
#define NT 10 		// Number of time samples per packet/mcnt
#define NM 4  		// Number of packets/mcnts per block
#define pow1 128 	// Next power of 2 >= Nm*Nt
#define pow2 64 	// Next power of 2 >= Nc

#ifdef __cplusplus
extern "C" {
#endif
void getTotalPower(unsigned char * input, float * output);
#ifdef __cplusplus
}
#endif

#endif
