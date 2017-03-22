#include "tools.h"


void genData(char* data, float* freq, float fs, int samples, int channels, int elements){


	fprintf(stdout,
			"INFO: Generating samples...\n"
			"\tSamples:\t %d\n"
			"\tSample rate:\t %f\n"
			"\tChannels:\t %d\n"
			"\tElements:\t %d\n",
			samples, fs, channels, elements);


	int n = 0;
	int f = 0;
	int e = 0;
 
	signed char dataRe = 0;
	signed char dataIm = 0;
	//int size = elements*channels*(2*sizeof(char)); // 2 for complex data.
	for(n = 0; n < samples; n++) {
		for(f = 0; f < channels; f++) {

			//if(f==5){ // only insert one tone

				//use the same sample for all elements
				dataRe = SCALE_FACTOR * (.1 * cos(2*M_PI * freq[f] * n / fs));
				dataIm = SCALE_FACTOR * (.1 * sin(2*M_PI * freq[f] * n / fs));
				for(e = 0; e < 2*elements; e++) {
					
					int idx = e + f * (2 * elements) + n * channels * (2*elements);
					if( !(e%2) ) {
					//create interleaved samples for real and Im 
					data[idx] = dataRe;
					} else {
					data[idx] = dataIm;
					}
				}
			//} else {
			//	cDataReX = 0;
			//	cDataImY = 0;
			//}
		}
	}
	return;
}