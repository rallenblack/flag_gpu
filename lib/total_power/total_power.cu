#include "total_power.h"

extern "C" void get_total_power(unsigned char * input, float * output);

__global__ void total_power_kernel1(unsigned char * input, float * output) {

    // Declare dynamic shared memory
    __shared__ float power[pow1];

    // Get indicies
    int c = blockIdx.x;
    int f = blockIdx.y;
    int t = threadIdx.x;
    
    // Get internal index (internal to block)
    int sid = t;

    // Get absolute index
    int idx = f + 8*Nf*c + 8*Nf*Nc*t;

    if (sid < Nm*Nt) {
    	// Extract real and imaginary components;
    	float real = (float)input[2*idx];
    	float imag = (float)input[2*idx + 1];

    	// Compute instantaneous power
    	power[sid] = real*real + imag*imag;
    }
    else {
    	power[sid] = 0;
    }

    // Complete power computation before moving on
    __syncthreads();

    // Perform reduction
    for (int s = blockDim.x/2; s > 0; s>>=1) {
        if (sid < s) {
            power[sid] += power[sid + s];
        }
        __syncthreads();
    }

    // Save sum to output
    if (sid == 0) {
        output[f + 8*Nf*c] = power[0];
    }
}


__global__ void total_power_kernel2(float * input, float * output) {

    // Declare dynamic shared memory
    __shared__ float power[pow2];
    
    // Get indices
    int f = blockIdx.x;
    int c = threadIdx.x;

    // Get internal index (internal to block)
    int sid = c;

    // Get absolute index
    int idx = f + 8*Nf*c;

    if (sid < Nc) {
    	// Copy input to shared memory
    	power[sid] = input[idx];
    }
    else {
    	power[sid] = 0.0;
    }

    // Finish copy before proceeding
    __syncthreads();

    // Perform reduction
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (sid < s) {
            power[sid] += power[sid + s];
        }
        __syncthreads();
    }

    // Save sum to input
    if (sid == 0) {
        output[f] = power[0];
    }
}


void get_total_power(unsigned char * input, float * output) {
    unsigned char * d_input;
    float * d_output1;
    float * d_output2;

    cudaMalloc((void **) &d_input, 8*Nf*Nc*Nt*Nm*sizeof(unsigned char)*2);
    cudaMalloc((void **) &d_output1, 8*Nf*Nc*sizeof(float));
    cudaMalloc((void **) &d_output2, 8*Nf*sizeof(float));
    cudaMemcpy(d_input, input, 8*Nf*Nc*Nt*Nm*2*sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 gridSize1(Nc,8*Nf,1);
    dim3 blockSize1(pow1,1);
    
    total_power_kernel1<<<gridSize1, blockSize1>>>(d_input, d_output1);
    cudaError_t ret = cudaGetLastError();
    if (ret != cudaSuccess) {
        printf("ERROR: total_power_kernel1 - %s\n", cudaGetErrorString(ret));
    }

    dim3 gridSize2(8*Nf,1,1);
    dim3 blockSize2(pow2,1,1);

    total_power_kernel2<<<gridSize2, blockSize2>>>(d_output1, d_output2);
    ret = cudaGetLastError();
    if (ret != cudaSuccess) {
        printf("ERROR: total_power_kernel2 - %s\n", cudaGetErrorString(ret));
    }

    cudaMemcpy(output, d_output2, 8*Nf*sizeof(float), cudaMemcpyDeviceToHost);
}
