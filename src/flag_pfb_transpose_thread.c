/* flag_pfb_transpose_thread.c
 * 
 * Routine to reorder received data to correct format for the PFB
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>

#include <xgpu.h>
#include "hashpipe.h"
#include "flag_databuf.h"

// Create thread status buffer
static hashpipe_status_t * st_p;

// Enumerated types for flag_transpose_thread state machine
typedef enum {
    ACQUIRE,
    CLEANUP
} state;

// Run method for the thread
// It is meant to do the following:
//     (1) Initialize status buffer
//     (2) Start main loop
//         (2a) Wait for input buffer block to be filled
//         (2b) Print out some data in the block
static void * run(hashpipe_thread_args_t * args) {
    // Local aliases to shorten access to args fields
    flag_input_databuf_t * db_in = (flag_input_databuf_t *)args->ibuf;
    flag_pfb_gpu_input_databuf_t * db_out = (flag_pfb_gpu_input_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    st_p = &st; // allow global (this source file) access to the status buffer

    // Mark thread as ready to run
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "TRAREADY", 1);
    hashpipe_status_unlock_safe(&st);
 
    // Set the default frequency chunk index
    int n_chunk = 0;
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "CHANSEL", n_chunk);
    hashpipe_status_unlock_safe(&st);

    int rv;
    int curblock_in = 0;
    int curblock_out = 0;
    uint64_t mcnt;
    state cur_state = ACQUIRE;
    state next_state = ACQUIRE;
    int traclean = -1;
    char netstat[17];
    while (run_threads()) {

        // Current state processing
        if (cur_state == ACQUIRE) {
            next_state = ACQUIRE;
            // Wait for input buffer block to be filled
            while ((rv=flag_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
                if (rv==HASHPIPE_TIMEOUT) { // If we are waiting for an input block...
                    // Check to see if network thread is in cleanup
                    hashpipe_status_lock_safe(&st);
                    hgetl(st.buf, "CLEANA", &traclean);
                    hgets(st.buf, "NETSTAT", 16, netstat);
                    hashpipe_status_unlock_safe(&st);
                    if (traclean == 0 && strcmp(netstat, "CLEANUP") == 0) {
                        next_state = CLEANUP;
                        break;
                    }
                }
                else {
                    hashpipe_error(__FUNCTION__, "error waiting for filled databuf block");
                    pthread_exit(NULL);
                    break;
                }
            }

            if (next_state != CLEANUP) {

                // Wait for output buffer block to be freed
                while ((rv=flag_pfb_gpu_input_databuf_wait_free(db_out, curblock_out)) != HASHPIPE_OK) {
                    if (rv == HASHPIPE_TIMEOUT) {
                        //hashpipe_status_lock_safe(&st);
                        //hputs(st.buf, status_key, "waiting for free block");
                        //hashpipe_status_unlock_safe(&st);
                        continue;
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
                hashpipe_status_lock_safe(&st);
                hputi4(st.buf, "TRAMCNT", mcnt);
                hashpipe_status_unlock_safe(&st);

                // Set metadata for output block
                db_out->block[curblock_out].header.mcnt = mcnt;
                db_out->block[curblock_out].header.good_data = tmp_header.good_data;
                //printf("TRA: Receiving block %d with starting mcnt = %lld\n", curblock_in, (long long int)mcnt);

                // Get the specified frequency channel chunk
                hashpipe_status_lock_safe(&st);
                hgeti4(st.buf, "CHANSEL", &n_chunk);
                hashpipe_status_unlock_safe(&st);
                int c_start = n_chunk*N_CHAN_PER_FRB_BLOCK;
                int c_end   = c_start + N_CHAN_PER_FRB_BLOCK;

                /**********************************************
                 * Perform transpose
                 **********************************************/
                int m; int f;
                int t; int c;
                uint64_t * in_p;
                uint64_t * out_p;
                uint64_t * block_in_p  = db_in->block[curblock_in].data;
		uint64_t * block_out_p = db_out->block[curblock_out].data;
                for (m = 0; m < Nm; m++) {
                    for (t = 0; t < Nt; t++) {
                        for (f = 0; f < Nf; f++) {
                            for (c = c_start; c < c_end; c++) {
                            // for (c = 0; c < Nc; c++) {
                                in_p  = block_in_p + flag_input_databuf_idx(m,f,t,c);
                                out_p = block_out_p + flag_gpu_input_databuf_idx(m,f,t,c % N_CHAN_PER_FRB_BLOCK);
                                memcpy(out_p, in_p, 128/8);
                            }
                        }
                    }
                }

    
                // Mark block as filled
                flag_pfb_gpu_input_databuf_set_filled(db_out, curblock_out);
                curblock_out = (curblock_out + 1) % db_out->header.n_block;
    
                // Set input block to free
                #if VERBOSE==1
                printf("TRA: Marking input block %d as free\n", curblock_in);
                #endif
                flag_input_databuf_set_free(db_in, curblock_in);
                curblock_in = (curblock_in + 1) % db_in->header.n_block;
            }
        }
        else if (cur_state == CLEANUP) {
            printf("TRA: In Clean up \n");
            curblock_in = 0;
            curblock_out = 0;
            next_state = ACQUIRE;
            // Indicate that we have finished cleanup
            hashpipe_status_lock_safe(&st);
            hputl(st.buf, "CLEANA", 1);
            hashpipe_status_unlock_safe(&st);
        }

        // Next state processing
        hashpipe_status_lock_safe(&st);
        switch(next_state) {
            case ACQUIRE: hputs(st.buf, status_key, "ACQUIRE"); break;
            case CLEANUP: hputs(st.buf, status_key, "CLEANUP"); break;
        }
        hashpipe_status_unlock_safe(&st);
        cur_state = next_state;
        pthread_testcancel();
    }

    // Thread terminates after loop
    return NULL;
}



// Thread description
static hashpipe_thread_desc_t t_thread = {
    name: "flag_pfb_transpose_thread",
    skey: "PTRASTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_input_databuf_create},
    obuf_desc: {flag_pfb_gpu_input_databuf_create}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&t_thread);
}

