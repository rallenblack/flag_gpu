#include "total_power.h"

__global__ void total_power_kernel1(unsigned char * input, float * output, int start_t) {

    // Declare dynamic shared memory
    __shared__ float power[nblocks1];

    // Get indicies
    int c = blockIdx.x;
    int f = blockIdx.y;
    int t = threadIdx.x;
    
    // Get internal index (internal to block)
    int sid = t;

    // Get absolute index
    int idx = f + 8*NF*c + 8*NF*NC*t;

    if (sid + start_t < NM*NT) {
    	// Extract real and imaginary components;
    	float real = (float)input[2*idx];
    	float imag = (float)input[2*idx + 1];

    	// Compute instantaneous power
    	power[sid] = real*real + imag*imag;
    }
    else {
    	power[sid] = 0.0;
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
        output[f + 8*NF*c] = power[0];
    }
}



__global__ void total_power_kernel2(float * input, float * output) {

    // Declare dynamic shared memory
    __shared__ float power[nblocks2];

    // Get indicies
    int c = blockIdx.x;
    int f = blockIdx.y;
    int t = threadIdx.x;
    
    // Get internal index (internal to block)
    int sid = t;

    // Get absolute index
    int idx = f + 8*NF*c + 8*NF*NC*t;

    power[sid] = input[idx];

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
        output[f + 8*NF*c] = power[0];
    }
}



__global__ void total_power_kernel3(float * input, float * output) {

    // Declare dynamic shared memory
    __shared__ float power[pow2];
    
    // Get indices
    int f = blockIdx.x;
    int c = threadIdx.x;

    // Get internal index (internal to block)
    int sid = c;

    // Get absolute index
    int idx = f + 8*NF*c;

    if (sid < NC) {
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

void getTotalPower(unsigned char * input, float * output) {
    unsigned char * d_input;
    float * d_output1;
    float * d_output2;
    float * d_output3;

    cudaMalloc((void **) &d_input, 8*NF*NC*NT*NM*sizeof(unsigned char)*2);
    cudaMalloc((void **) &d_output1, 8*NF*NC*nblocks2*sizeof(float));
    cudaMalloc((void **) &d_output2, 8*NF*NC*sizeof(float));
    cudaMalloc((void **) &d_output3, 8*NF*sizeof(float));
    cudaMemcpy(d_input, input, 8*NF*NC*NT*NM*2*sizeof(unsigned char), cudaMemcpyHostToDevice);


    /**********************************************
     * Reduce over 1024 time samples
     **********************************************/
    dim3 gridSize1(NC,8*NF,1);
    dim3 blockSize1(nblocks1,1,1);
    
    for (int i = 0; i < nblocks2; i++) {
        int in_off = 8*NF*NC*nblocks1*i;
        int out_off = 8*NF*NC*i;
        total_power_kernel1<<<gridSize1, blockSize1>>>(d_input+in_off, d_output1+out_off, nblocks1*i);
    }
    cudaError_t ret = cudaGetLastError();
    if (ret != cudaSuccess) {
        printf("ERROR: total_power_kernel1 - %s\n", cudaGetErrorString(ret));
    }




    /**********************************************
     * Reduce over remaining time samples
     **********************************************/
    dim3 gridSize2(NC,8*NF,1);
    dim3 blockSize2(nblocks2,1,1);
    total_power_kernel2<<<gridSize2, blockSize2>>>(d_output1, d_output2);
    ret = cudaGetLastError();
    if (ret != cudaSuccess) {
        printf("ERROR: total_power_kernel2 - %s\n", cudaGetErrorString(ret));
    }


    /**********************************************
     * Reduce over frequency channels
     **********************************************/
    dim3 gridSize3(8*NF,1,1);
    dim3 blockSize3(pow2,1,1);

    total_power_kernel3<<<gridSize3, blockSize3>>>(d_output2, d_output3);
    ret = cudaGetLastError();
    if (ret != cudaSuccess) {
        printf("ERROR: total_power_kernel3 - %s\n", cudaGetErrorString(ret));
    }

    cudaMemcpy(output, d_output3, 8*NF*sizeof(float), cudaMemcpyDeviceToHost);
}
