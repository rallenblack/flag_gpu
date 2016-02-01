#ifndef TOTAL_POWER_H
#define TOTAL_POWER_H

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define Nf 5 		// Number of Fengines
#define Ni 8 		// Number of inputs per Fengine
#define Na (Nf*Ni) 	// Number of total antennas
#define Nc 50 		// Number of frequency channels
#define Nt 10 		// Number of time samples per packet/mcnt
#define Nm 4  		// Number of packets/mcnts per block
#define pow1 128 	// Next power of 2 >= Nm*Nt
#define pow2 64 	// Next power of 2 >= Nc

#endif
