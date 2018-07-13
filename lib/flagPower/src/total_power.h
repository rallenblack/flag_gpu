#ifndef TOTAL_POWER_H
#define TOTAL_POWER_H

#include <stdlib.h>
#include <stdio.h>

#define NF 8 		// Number of Fengines
#define NI 8 		// Number of inputs per Fengine
#define NA (NF*NI) 	// Number of total antennas
#define NC 20 // 25 		// Number of frequency channels
#define NT 20 		// Number of time samples per packet/mcnt
#define NM 400 // 200 	// Number of packets/mcnts per block
#define pow1 32768 	// Next power of 2 >= Nm*Nt
#define nblocks2 (pow1/1024) // Block size for second kernel
#define nblocks1 (pow1/nblocks2) // Block size for first kernel
#define pow2 32 	// Next power of 2 >= Nc

#ifdef __cplusplus
extern "C" {
#endif
void initTotalPower();
void getTotalPower(unsigned char * input, float * output);
#ifdef __cplusplus
}
#endif

#endif
