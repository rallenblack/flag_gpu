#include "total_power.h"

int main() {
    unsigned char data[8*Nf*Nc*Nt*Nm*2];
    float output[8*Nf];

    int f;
    int c;
    int t;
    for (f = 0; f < 8*Nf; f++) {
    	for (c = 0; c < Nc; c++) {
    		for (t = 0; t < Nt*Nm; t++) {
    			if (f == 1) {
    				data[2*(f + c*8*Nf + t*8*Nf*Nc)] = 0;
    				data[2*(f + c*8*Nf + t*8*Nf*Nc) + 1] = 1;
    			}
    			else {
    				data[2*(f + c*8*Nf + t*8*Nf*Nc)] = 0;
    				data[2*(f + c*8*Nf + t*8*Nf*Nc) + 1] = 0;
    			}
    		}
    	}
    }

    get_total_power(data, output);
    int a;
    for (a = 0; a < 8*Nf; a++) {
        printf("el %d = %f\n", a, output[a]);
    }

    return 0;
}
