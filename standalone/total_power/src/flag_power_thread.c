/* flag__power_thread.c
 * 
 * Routine to compute total power from received packets
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#include "total_power.h"
#include "hashpipe.h"
#include "flag_databuf.h"

// Create thread status buffer
static hashpipe_status_t * st_p;


// Run method for the thread
// It is meant to do the following:
//     (1) Initialize status buffer
//     (2) Start main loop
//         (2a) Wait for input buffer block to be filled
//         (2b) Print out some data in the block
static void * run(hashpipe_thread_args_t * args) {
    // Local aliases to shorten access to args fields
    flag_gpu_input_databuf_t * db_in = (flag_gpu_input_databuf_t *)args->ibuf;
    flag_gpu_output_databuf_t * db_out = (flag_gpu_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    st_p = &st; // allow global (this source file) access to the status buffer

    int rv;
    int curblock_in = 0;
    int curblock_out = 0;
    uint64_t start_mcnt = 0;
    uint64_t last_mcnt = Nm - 1;

    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "CORREADY", 1);
    hashpipe_status_unlock_safe(&st);
    while (run_threads()) {
        
	// Wait for input buffer block to be filled
        while ((rv=flag_gpu_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "waiting for free block");
                hashpipe_status_unlock_safe(&st);	
            }
            else {
                hashpipe_error(__FUNCTION__, "error waiting for filled databuf block");
                pthread_exit(NULL);
                break;
            }
        }

        // Print out the header information for this block 
        flag_gpu_input_header_t tmp_header;
        memcpy(&tmp_header, &db_in->block[curblock_in].header, sizeof(flag_gpu_input_header_t));
	printf("TOT: Received block %d, starting mcnt = %lld\n", curblock_in, (long long int)tmp_header.mcnt);

        while ((rv=flag_gpu_output_databuf_wait_free(db_out, curblock_out)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                continue;
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                fprintf(stderr, "rv = %d\n", rv);
                pthread_exit(NULL);
                break;
            }
        }

        printf("TOT: Output block %d free\n", curblock_out);
        
       
        //xgpuCudaXengine(&context, doDump ? SYNCOP_DUMP : SYNCOP_SYNC_TRANSFER);
        getTotalPower((unsigned char *)&db_in->block[curblock_in].data, (float *)&db_out->block[curblock_out].data);
        
        db_out->block[curblock_out].header.mcnt = start_mcnt;
            
        // Mark output block as full and advance
        flag_gpu_output_databuf_set_filled(db_out, curblock_out);
        curblock_out = (curblock_out + 1) % db_out->header.n_block;
        start_mcnt = last_mcnt + 1;
        last_mcnt = start_mcnt + Nm - 1;
        
        // Mark input block as free
        flag_gpu_input_databuf_set_free(db_in, curblock_in);
        curblock_in = (curblock_in + 1) % db_in->header.n_block;
        pthread_testcancel();
    }

    // Thread terminates after loop
    return NULL;
}



// Thread description
static hashpipe_thread_desc_t x_thread = {
    name: "flag_power_thread",
    skey: "TOTSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_gpu_input_databuf_create},
    obuf_desc: {flag_gpu_output_databuf_create}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&x_thread);
}

