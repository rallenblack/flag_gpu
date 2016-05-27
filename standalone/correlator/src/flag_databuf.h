#ifndef _FLAG_DATABUF_H
#define _FLAG_DATABUF_H

#include <stdint.h>
#include "hashpipe_databuf.h"
#include "config.h"


// Total number of antennas (nominally 40)
// Determined by XGPU code
#define N_INPUTS (2*XGPU_NSTATION)
// xGPU needs a multiple of 32 inputs. The real number is...
#define N_REAL_INPUTS 40

// Number of antennas per F engine
// Determined by F engine DDL cards
#define N_INPUTS_PER_FENGINE 8

// Number of F engines
#define N_FENGINES (N_INPUTS/N_INPUTS_PER_FENGINE)
#define N_REAL_FENGINES (N_REAL_INPUTS/N_INPUTS_PER_FENGINE)

// Number of X engines
#define N_XENGINES (20)

// Number of inputs per packet
#define N_INPUTS_PER_PACKET N_INPUTS_PER_FENGINE

// Number of time samples per packet
#define N_TIME_PER_PACKET 20

// Number of bits per I/Q sample
// Determined by F engine packetizer
#define N_BITS_IQ 8

// Number of channels in system
#define N_CHAN_TOTAL 512

// Number of throwaway channels
#define N_CHAN_THROWAWAY 12

// Total number of processed channels
#define N_CHAN (N_CHAN_TOTAL - N_CHAN_THROWAWAY)

// Number of channels per packet
#define N_CHAN_PER_PACKET (N_CHAN/N_XENGINES)

// Number of channels processed per XGPU instance?
#define N_CHAN_PER_X XGPU_NFREQUENCY

// Number of time samples processed per XGPU instance?
#define N_TIME_PER_BLOCK XGPU_NTIME

// Number of bytes per packet
#define N_BYTES_PER_PACKET ((N_BITS_IQ * 2)*N_INPUTS_PER_FENGINE*N_CHAN_PER_PACKET/8*N_TIME_PER_PACKET + 8)

// Number of bytes in packet payload
#define N_BYTES_PER_PAYLOAD (N_BYTES_PER_PACKET - 8)

// Number of bytes per block
#define N_BYTES_PER_BLOCK (N_TIME_PER_BLOCK * N_CHAN_PER_X * N_INPUTS * N_BITS_IQ * 2 / 8)
#define N_REAL_BYTES_PER_BLOCK (N_TIME_PER_BLOCK * N_CHAN_PER_X * N_REAL_INPUTS * N_BITS_IQ * 2 / 8)
// #define N_BYTES_PER_BLOCK (N_TIME_PER_BLOCK * N_CHAN_PER_PACKET * N_INPUTS)

// Number of packets per block
#define N_PACKETS_PER_BLOCK (N_BYTES_PER_BLOCK / N_BYTES_PER_PAYLOAD)
#define N_REAL_PACKETS_PER_BLOCK (N_REAL_BYTES_PER_BLOCK / N_BYTES_PER_PAYLOAD)

// Macro to compute data word offset for complex data word
#define Nm (N_TIME_PER_BLOCK/N_TIME_PER_PACKET) // Number of mcnts per block
#define Nf (N_FENGINES) // Number of fengines
#define Nt (N_TIME_PER_PACKET) // Number of time samples per packet
#define Nc (N_CHAN_PER_PACKET) // Number of channels per packet
#define flag_input_databuf_idx(m,f,t,c) ((2*N_INPUTS_PER_FENGINE/sizeof(uint64_t))*(c+Nc*(t+Nt*(f+Nf*m))))
//#define flag_input_databuf_idx(m,f,t,c) ((2*N_INPUTS_PER_FENGINE/sizeof(uint64_t))*(c+Nc*(t+Nt*(f+Nf*m))))

// Macro to compute data word offset for transposed matrix
//#define flag_gpu_input_databuf_idx(m,f,t,c) ((2*N_INPUTS_PER_FENGINE/sizeof(uint64_t))*(c+Nc*(f+Nf*(t+Nt*m))))
#define flag_gpu_input_databuf_idx(m,f,t,c) ((2*N_INPUTS_PER_FENGINE/sizeof(uint64_t))*(f+Nf*(c+Nc*(t+Nt*m))))

// Number of entries in output correlation matrix
// #define N_COR_MATRIX (N_INPUTS*(N_INPUTS + 1)/2*N_CHAN_PER_X)
#define N_COR_MATRIX (N_INPUTS/2*(N_INPUTS/2 + 1)/2*N_CHAN_PER_X*4)


// Macros to maintain cache alignment
#define CACHE_ALIGNMENT (128)
typedef uint8_t hashpipe_databuf_cache_alignment[
    CACHE_ALIGNMENT - (sizeof(hashpipe_databuf_t)%CACHE_ALIGNMENT)
];

/*
 * INPUT (NET) BUFFER STRUCTURES
 * This buffer is where captured data from the network is stored.
 * It is the output buffer of the flag_net_thread.
 * It is the input buffer of the flag_transpose_thread.
 */
#define N_INPUT_BLOCKS 10

// A typedef for a block header
typedef struct flag_input_header {
    int64_t  good_data;
    uint64_t mcnt_start;
} flag_input_header_t;

typedef uint8_t flag_input_header_cache_alignment[
    CACHE_ALIGNMENT - (sizeof(flag_input_header_t)%CACHE_ALIGNMENT)
];

// A typedef for a block of data in the buffer
typedef struct flag_input_block {
    flag_input_header_t header;
    flag_input_header_cache_alignment padding;
    uint64_t data[N_BYTES_PER_BLOCK/sizeof(uint64_t)];
} flag_input_block_t;

// The data buffer structure
typedef struct flag_input_databuf {
    hashpipe_databuf_t header;
    hashpipe_databuf_cache_alignment padding; // Only to maintain alignment
    flag_input_block_t block[N_INPUT_BLOCKS];
} flag_input_databuf_t;

/*
 * GPU INPUT BUFFER STRUCTURES
 * This buffer is where the reordered data for input to xGPU is stored.
 * It is the output buffer of the flag_transpose_thread.
 * It is the input buffer of the flag_correlator_thread.
 */
#define N_GPU_INPUT_BLOCKS 2

// A typedef for a GPU input block header
typedef struct flag_gpu_input_header {
    uint64_t mcnt;
} flag_gpu_input_header_t;

typedef uint8_t flag_gpu_input_header_cache_alignment[
    CACHE_ALIGNMENT - (sizeof(flag_gpu_input_header_t)%CACHE_ALIGNMENT)
];

// A typedef for a block of data in the buffer
typedef struct flag_gpu_input_block {
    flag_gpu_input_header_t header;
    flag_gpu_input_header_cache_alignment padding;
    uint64_t data[N_BYTES_PER_BLOCK/sizeof(uint64_t)];
} flag_gpu_input_block_t;

// The data buffer structure
typedef struct flag_gpu_input_databuf {
    hashpipe_databuf_t header;
    hashpipe_databuf_cache_alignment padding;
    flag_gpu_input_block_t block[N_GPU_INPUT_BLOCKS];
} flag_gpu_input_databuf_t;


/*
 * CORRELATOR OUTPUT BUFFER STRUCTURES
 */
#define N_COR_OUT_BLOCKS 2

// A typedef for a correlator output block header
typedef struct flag_correlator_output_header {
    uint64_t mcnt;
    uint64_t flags[(N_CHAN_PER_X+63)/64];
} flag_correlator_output_header_t;

typedef uint8_t flag_correlator_output_header_cache_alignment[
    CACHE_ALIGNMENT - (sizeof(flag_correlator_output_header_t)%CACHE_ALIGNMENT)
];

// A typedef for a block of data in the GPU output buffer
typedef struct flag_correlator_output_block {
    flag_correlator_output_header_t header;
    flag_correlator_output_header_cache_alignment padding;
    float data[2*N_COR_MATRIX];
} flag_correlator_output_block_t;

// A typedef for the data buffer structure
typedef struct flag_correlator_output_databuf {
    hashpipe_databuf_t header;
    hashpipe_databuf_cache_alignment padding;
    flag_correlator_output_block_t block[N_COR_OUT_BLOCKS];
} flag_correlator_output_databuf_t;


/*********************
 * Input Buffer Functions
 *********************/
hashpipe_databuf_t * flag_input_databuf_create(int instance_id, int databuf_id);

int flag_input_databuf_wait_free   (flag_input_databuf_t * d, int block_id);
int flag_input_databuf_wait_filled (flag_input_databuf_t * d, int block_id);
int flag_input_databuf_set_free    (flag_input_databuf_t * d, int block_id);
int flag_input_databuf_set_filled  (flag_input_databuf_t * d, int block_id);

/*********************
 * Input Buffer Functions
 *********************/
hashpipe_databuf_t * flag_gpu_input_databuf_create(int instance_id, int databuf_id);
int flag_gpu_input_databuf_wait_free   (flag_gpu_input_databuf_t * d, int block_id);
int flag_gpu_input_databuf_wait_filled (flag_gpu_input_databuf_t * d, int block_id);
int flag_gpu_input_databuf_set_free    (flag_gpu_input_databuf_t * d, int block_id);
int flag_gpu_input_databuf_set_filled  (flag_gpu_input_databuf_t * d, int block_id);

/********************
 * Correlator Output Buffer Functions
 ********************/
hashpipe_databuf_t * flag_correlator_output_databuf_create(int instance_id, int databuf_id);

int flag_correlator_output_databuf_wait_free   (flag_correlator_output_databuf_t * d, int block_id);
int flag_correlator_output_databuf_wait_filled (flag_correlator_output_databuf_t * d, int block_id);
int flag_correlator_output_databuf_set_free    (flag_correlator_output_databuf_t * d, int block_id);
int flag_correlator_output_databuf_set_filled  (flag_correlator_output_databuf_t * d, int block_id);



#endif
