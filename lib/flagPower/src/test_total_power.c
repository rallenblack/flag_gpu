#include "total_power.h"

int main() {
    unsigned char data[8*NF*NC*NT*NM*2];
    float output[8*NF];

    int f;
    int c;
    int t;
    for (f = 0; f < 8*NF; f++) {
    	for (c = 0; c < NC; c++) {
    		for (t = 0; t < NT*NM; t++) {
    			if (f == 1) {
    				data[2*(f + c*8*NF + t*8*NF*NC)] = 0;
    				data[2*(f + c*8*NF + t*8*NF*NC) + 1] = 1;
    			}
    			else {
    				data[2*(f + c*8*NF + t*8*NF*NC)] = 0;
    				data[2*(f + c*8*NF + t*8*NF*NC) + 1] = 0;
    			}
    		}
    	}
    }

    getTotalPower(data, output);
    int a;
    for (a = 0; a < 8*NF; a++) {
        printf("el %d = %f\n", a, output[a]);
    }

    return 0;
}
