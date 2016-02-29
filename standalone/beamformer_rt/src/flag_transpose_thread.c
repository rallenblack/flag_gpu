/* flag_transpose_thread.c
 * 
 * Routine to reorder received data to correct format for xGPU
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#include <xgpu.h>
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
    flag_input_databuf_t * db_in = (flag_input_databuf_t *)args->ibuf;
    flag_gpu_input_databuf_t * db_out = (flag_gpu_input_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    st_p = &st; // allow global (this source file) access to the status buffer

    // Mark thread as ready to run
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "TRAREADY", 1);
    hashpipe_status_unlock_safe(&st);
 
    int rv;
    int curblock_in = 0;
    int curblock_out = 0;
    int mcnt;
    char integ_status[17];
    while (run_threads()) {
        
        // Wait for input buffer block to be filled
        while ((rv=flag_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "waiting for filled block");
                hashpipe_status_unlock_safe(&st);
            }
            else {
                hashpipe_error(__FUNCTION__, "error waiting for filled databuf block");
                pthread_exit(NULL);
                break;
            }
            hashpipe_status_lock(&st);
	    hgets(st.buf, "INTSTAT", 16, integ_status);
	    hashpipe_status_unlock(&st);
	    if (strcmp(integ_status, "stop") == 0) {
		curblock_in = 0;
	    }
        }

        // Wait for output buffer block to be freed
        while ((rv=flag_gpu_input_databuf_wait_free(db_out, curblock_out)) != HASHPIPE_OK) {
            if (rv == HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(&st);
                hputs(st.buf, status_key, "waiting for free block");
                hashpipe_status_unlock_safe(&st);
            }
            else {
                hashpipe_error(__FUNCTION__, "error waiting for free databuf block");
                pthread_exit(NULL);
                break;
            }
        }
       
        // Print out the header information for this block 
        flag_input_header_t tmp_header;
        memcpy(&tmp_header, &db_in->block[curblock_in].header, sizeof(flag_input_header_t));
        mcnt = tmp_header.mcnt_start;
	// printf("TRA: Receiving mcnt = %lld\n", (long long int)mcnt);

        // Perform transpose

        int m; int f;
        int t; int c;
        uint64_t * in_p;
        uint64_t * out_p;
        uint64_t * block_in_p  = db_in->block[curblock_in].data;
        uint64_t * block_out_p = db_out->block[curblock_out].data;
        for (m = 0; m < Nm; m++) {
            for (t = 0; t < Nt; t++) {
                for (f = 0; f < Nf; f++) {
                    for (c = 0; c < Nc; c++) {
                        in_p  = block_in_p + flag_input_databuf_idx(m,f,t,c);
                        out_p = block_out_p + flag_gpu_input_databuf_idx(m,f,t,c);
                        //fprintf(stderr, "(m,t,f,c) = (%d,%d,%d,%d), in_off = %lu, out_off = %lu\n", m, t, f, c, flag_input_databuf_idx(m,f,t,c), flag_gpu_input_databuf_idx(m,f,t,c));
                        memcpy(out_p, in_p, 128/8);
                    }
                }
            }
        }
        db_out->block[curblock_out].header.mcnt = mcnt;

        flag_gpu_input_databuf_set_filled(db_out, curblock_out);
        curblock_out = (curblock_out + 1) % db_out->header.n_block;

        flag_input_databuf_set_free(db_in, curblock_in);
        curblock_in = (curblock_in + 1) % db_in->header.n_block;

        pthread_testcancel();
    }

    // Thread terminates after loop
    return NULL;
}



// Thread description
static hashpipe_thread_desc_t t_thread = {
    name: "flag_transpose_thread",
    skey: "CORSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_input_databuf_create},
    obuf_desc: {flag_gpu_input_databuf_create}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&t_thread);
}

