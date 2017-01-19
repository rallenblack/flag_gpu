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
    flag_gpu_input_databuf_t * db_in = (flag_gpu_input_databuf_t *)args->ibuf;
    flag_gpu_power_output_databuf_t * db_out = (flag_gpu_power_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    st_p = &st; // allow global (this source file) access to the status buffer

    initTotalPower();

    int rv;
    int curblock_in = 0;
    int curblock_out = 0;
    uint64_t start_mcnt = 0;
    uint64_t last_mcnt = Nm - 1;

    state cur_state = ACQUIRE;
    state next_state = ACQUIRE;
    int64_t good_data = 1;
    char netstat[17];

    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "CORREADY", 1);
    hashpipe_status_unlock_safe(&st);
    while (run_threads()) {
        
        if (cur_state == ACQUIRE) {
            next_state = ACQUIRE;

		    // Wait for input buffer block to be filled
		    while ((rv=flag_gpu_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
		        if (rv==HASHPIPE_TIMEOUT) {
                    int cleanb;
                    hashpipe_status_lock_safe(&st);
                    hgetl(st.buf, "CLEANB", &cleanb);
                    hgets(st.buf, "NETSTAT", 16, netstat);
                    hashpipe_status_unlock_safe(&st);
                    if (cleanb == 0 && strcmp(netstat, "CLEANUP") == 0) {
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

		    // Print out the header information for this block 
		    flag_gpu_input_header_t tmp_header;
		    memcpy(&tmp_header, &db_in->block[curblock_in].header, sizeof(flag_gpu_input_header_t));
            good_data = tmp_header.good_data;
                    //printf("TOT: Got block %d (mcnt = %lld)\n", curblock_in, (long long int)start_mcnt);

		    while ((rv=flag_gpu_power_output_databuf_wait_free(db_out, curblock_out)) != HASHPIPE_OK) {
		        if (rv==HASHPIPE_TIMEOUT) {
		            continue;
		        } else {
		            hashpipe_error(__FUNCTION__, "error waiting for free databuf");
		            fprintf(stderr, "rv = %d\n", rv);
		            pthread_exit(NULL);
		            break;
		        }
		    }

		    getTotalPower((unsigned char *)&db_in->block[curblock_in].data, (float *)&db_out->block[curblock_out].data);
		
                    db_out->block[curblock_out].header.mcnt = start_mcnt;
                    db_out->block[curblock_out].header.good_data = good_data;

                    //printf("TOT: Marking block %d (mcnt = %lld) as filled\n", curblock_out, (long long int)start_mcnt);
		        
		    // Mark output block as full and advance
		    flag_gpu_power_output_databuf_set_filled(db_out, curblock_out);
		    curblock_out = (curblock_out + 1) % db_out->header.n_block;
		    start_mcnt = last_mcnt + 1;
		    last_mcnt = start_mcnt + Nm - 1;
		
		    // Mark input block as free
		    flag_gpu_input_databuf_set_free(db_in, curblock_in);
		    curblock_in = (curblock_in + 1) % db_in->header.n_block;
        }
        else if (cur_state == CLEANUP) {
            next_state = ACQUIRE;
            curblock_in = 0;
            curblock_out = 0;
            hashpipe_status_lock_safe(&st);
            hputl(st.buf, "CLEANB", 1);
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
static hashpipe_thread_desc_t pow_thread = {
    name: "flag_power_thread",
    skey: "TOTSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_gpu_input_databuf_create},
    obuf_desc: {flag_gpu_power_output_databuf_create}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&pow_thread);
}

