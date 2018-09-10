/* flag_frb_correlator_thread.c
 * 
 * Routine to correlate received packets for FRB searches
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

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
    flag_frb_gpu_input_databuf_t * db_in = (flag_frb_gpu_input_databuf_t *)args->ibuf;
    flag_frb_gpu_correlator_output_databuf_t * db_out = (flag_frb_gpu_correlator_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    st_p = &st; // allow global (this source file) access to the status buffer

    // Initialize correlator integrator status to "off"
    // Initialize starting mcnt to 0 (INTSYNC)
    char integ_status[17];
    int gpu_dev = 0;
    hashpipe_status_lock_safe(&st);
    hputs(st.buf, "INTSTAT", "off");
    hputi8(st.buf, "INTSYNC", 0);
    hputr4(st.buf, "REQSTI", 0.0001); // Requested STI length (set by Dealer/Player)
    hputr4(st.buf, "ACTSTI", 0.0); // Delivered (actual) STI length (based on whole number of blocks)
    hputi4(st.buf, "INTCOUNT", 1); // Number of blocks to integrate per STI
    hgeti4(st.buf, "GPUDEV", &gpu_dev);
    hputi4(st.buf, "GPUDEV", gpu_dev);
    hashpipe_status_unlock_safe(&st);

    // Initialize xGPU context structure
    // Comment from PAPER:
    //   Initialize context to point at first input and output memory blocks.
    //   This seems redundant since we do this just before calling
    //   xgpuCudaXengine, but we need to pass something in for array_h and
    //   matrix_x to prevent xgpuInit from allocating memory.
    XGPUContext context;
    context.array_h  = (ComplexInput *)db_in->block[0].data;
    context.matrix_h = (Complex *)db_out->block[0].data;
   
    context.array_len = (db_in->header.n_block * sizeof(flag_frb_gpu_input_block_t) - sizeof(flag_gpu_input_header_t))/sizeof(ComplexInput);
    context.matrix_len = (db_out->header.n_block * sizeof(flag_frb_gpu_correlator_output_block_t) - sizeof(flag_gpu_output_header_t))/sizeof(Complex);

    int xgpu_error = xgpuInit(&context, gpu_dev);
    if (XGPU_OK != xgpu_error) {
        fprintf(stderr, "ERROR: xGPU initialization failed (error code %d)\n", xgpu_error);
        return THREAD_ERROR;
    }

    #if VERBOSE==1
        printf("XGPU CONTEXT\n");
        printf("############################################################\n");
        printf("header.n_block = %lld\n", (long long int)(db_in->header.n_block));
        printf("sizeof(flag_frb_gpu_input_block_t) = %lld\n", (long long int)(sizeof(flag_frb_gpu_input_block_t)));
        printf("sizeof(flag_gpu_input_header_t) = %lld\n", (long long int)(sizeof(flag_gpu_input_header_t)));
        printf("sizeof(flag_frb_gpu_correlator_output_block_t) = %d\n", (int)(sizeof(flag_gpu_correlator_output_block_t)));
        printf("sizeof(flag_gpu_output_header_t) = %lld\n", (long long int)(sizeof(flag_gpu_output_header_t)));
        printf("sizeof(ComplexInput) = %d\n", (int)(sizeof(ComplexInput)));
        printf("sizeof(Complex) = %d\n", (int)(sizeof(Complex)));
        printf("context.array_len  = %lld\n", (long long int)(context.array_len));
        printf("context.matrix_len = %lld\n", (long long int)(context.matrix_len));
        printf("###########################################################\n");
    #endif

    #if VERBOSE==1
        printf("N_TIME_PER_FRB_BLOCK = %d\n", N_TIME_PER_FRB_BLOCK);
        printf("N_CHAN_PER_FRB_BLOCK = %d\n", N_CHAN_PER_FRB_BLOCK);
        printf("N_GPU_FRB_INPUT_BLOCKS = %d\n", N_GPU_FRB_INPUT_BLOCKS);
        printf("N_FRB_COR_MATRIX = %d\n", N_FRB_COR_MATRIX);
        printf("N_BYTES_PER_FRB_BLOCK = %lld\n", (long long int) N_BYTES_PER_FRB_BLOCK);
    #endif
    

    // Mark thread as ready to run
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "CORREADY", 1);
    hashpipe_status_unlock_safe(&st);
 
    int rv;
    int curblock_in = 0;
    int curblock_out = 0;
    uint64_t start_mcnt = 0;
    uint64_t last_mcnt = 0;
    int int_count = 1; // Number of blocks to integrate per dump
    state cur_state = ACQUIRE;
    state next_state = ACQUIRE;
    char netstat[17];
    int64_t good_data = 1;
    while (run_threads()) {
       
        if (cur_state == ACQUIRE) {
            next_state = ACQUIRE;
	    // Wait for input buffer block to be filled
            while ((rv=flag_frb_gpu_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
                if (rv==HASHPIPE_TIMEOUT) {
                    hashpipe_status_lock_safe(&st);
                    hputs(st.buf, "FRB", "stuck waiting for data");
                    hashpipe_status_unlock_safe(&st);

                    int cleanb;
                    hashpipe_status_lock_safe(&st);
                    hgetl(st.buf, "CLEANB", &cleanb);
                    hgets(st.buf, "NETSTAT", 16, netstat);
                    hashpipe_status_unlock_safe(&st);
                    if (cleanb == 0 && strcmp(netstat, "CLEANUP") == 0) {
                       printf("COR: Cleanup condition met!\n");
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
            // Print out the header information for this block 
            flag_gpu_input_header_t tmp_header;
            memcpy(&tmp_header, &db_in->block[curblock_in].header, sizeof(flag_gpu_input_header_t));
            #if VERBOSE==1
    	        printf("COR: Received block %d, starting mcnt = %lld\n", curblock_in, (long long int)tmp_header.mcnt);
	    #endif
            good_data &= tmp_header.good_data;
            hashpipe_status_lock_safe(&st);
            hputi4(st.buf, "CORMCNT", tmp_header.mcnt);
            hashpipe_status_unlock_safe(&st);

            // Retrieve correlator integrator status
            hashpipe_status_lock_safe(&st);
            hgets(st.buf, "INTSTAT", 16, integ_status);
            hashpipe_status_unlock_safe(&st);
        
            // If the correlator integrator status is "off,"
            // Free the input block and continue
            if (strcmp(integ_status, "off") == 0) {
                #if VERBOSE==1
                    fprintf(stderr, "COR: Correlator is off...\n");
                #endif
                flag_frb_gpu_input_databuf_set_free(db_in, curblock_in);
                curblock_in = (curblock_in + 1) % db_in->header.n_block;
                good_data = 1;
                continue;
            }

            // If the correlator integrator status is "start,"
            // Get the correlator started
            // The INTSTAT string is set to "start" by the net thread once it's up and running
            if (strcmp(integ_status, "start") == 0) {

	        // Get the starting mcnt for integration (should be zero)
                hashpipe_status_lock_safe(&st);
                hgeti4(st.buf, "NETMCNT", (int *)(&start_mcnt));
                hashpipe_status_unlock_safe(&st); 

                // Check to see if block's starting mcnt matches INTSYNC
                if (db_in->block[curblock_in].header.mcnt < start_mcnt) {

		    // If we get here, then there is a bug since the net thread shouldn't
		    // mark blocks as filled that are before the starting mcnt
                    // fprintf(stderr, "COR: Unable to start yet... waiting for mcnt = %lld\n", (long long int)start_mcnt);

                    // starting mcnt not yet reached
                    // free block and continue
                    flag_frb_gpu_input_databuf_set_free(db_in, curblock_in);
                    curblock_in = (curblock_in + 1) % db_in->header.n_block;
                    continue;
                }
                else if (db_in->block[curblock_in].header.mcnt == start_mcnt) {
                    // set correlator integrator to "on"
                    #if VERBOSE==1
                        fprintf(stderr, "COR: Starting correlator!\n");
                    #endif
                    strcpy(integ_status, "on");
                    float requested_integration_time = 0.0;
                    float actual_integration_time = 0.0;
                    hashpipe_status_lock_safe(&st);
                    hputs(st.buf, "INTSTAT", integ_status);
                    hgetr4(st.buf, "REQSTI", &requested_integration_time);
                    hashpipe_status_unlock_safe(&st);
                    printf("COR: Requested integration length = %f\n", requested_integration_time);

                    int_count = ceil((N_MCNT_PER_SECOND / N_MCNT_PER_FRB_BLOCK) * requested_integration_time);
                    actual_integration_time = int_count/(N_MCNT_PER_SECOND / N_MCNT_PER_FRB_BLOCK);

                    hashpipe_status_lock_safe(&st);
                    hputr4(st.buf, "ACTSTI", actual_integration_time);
                    hputi4(st.buf, "INTCOUNT", int_count);
                    hashpipe_status_unlock_safe(&st);

                    // Compute last mcount
                    last_mcnt = start_mcnt + int_count*N_MCNT_PER_FRB_BLOCK - 1;
                }
                else {
                    // fprintf(stdout, "COR: We missed the start of the integration\n");
		    fprintf(stdout, "COR: Missed start. Expected start_mcnt = %lld, got %lld\n", (long long int)start_mcnt, (long long int)db_in->block[curblock_in].header.mcnt);
                    // we apparently missed the start of the integation... ouch...
                }
            }

	    // Check to see if a stop is issued
	    if (strcmp(integ_status, "stop") == 0) {
	        continue;
	    }

            // If we get here, then integ_status == "on"
            // Setup for current chunk
            context.input_offset  = curblock_in  * sizeof(flag_frb_gpu_input_block_t) / sizeof(ComplexInput);
            context.output_offset = curblock_out * sizeof(flag_frb_gpu_correlator_output_block_t) / sizeof(Complex);
        
            int doDump = 0;
            if ((db_in->block[curblock_in].header.mcnt + int_count*N_MCNT_PER_FRB_BLOCK - 1) >= last_mcnt) {
                doDump = 1;

                // Wait for new output block to be free
                while ((rv=flag_frb_gpu_correlator_output_databuf_wait_free(db_out, curblock_out)) != HASHPIPE_OK) {
                    if (rv==HASHPIPE_TIMEOUT) {
                        int cleanb;
                        hashpipe_status_lock_safe(&st);
                        hgetl(st.buf, "CLEANB", &cleanb);
                        hgets(st.buf, "NETSTAT", 16, netstat);
                        hashpipe_status_unlock_safe(&st);
                        if (cleanb == 0 && strcmp(netstat, "CLEANUP") == 0) {
                           printf("COR: Cleanup condition met!\n");
                           next_state = CLEANUP;
                           break;
                        }
                        continue;
                    } else {
                        hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                        // fprintf(stderr, "rv = %d\n", rv);
                        pthread_exit(NULL);
                        break;
                    }
                }
            }
       
            #if VERBOSE==1
                printf("COR: Running xgpuCudaXengine now...\n");
            #endif
            xgpuCudaXengine(&context, doDump ? SYNCOP_DUMP : SYNCOP_SYNC_TRANSFER);
            #if VERBOSE==1
                printf("COR: Done!\n");
            #endif
       
            #if VERBOSE==1 
                printf("COR: doDump = %d\n", doDump);
                printf("COR: start_mcnt = %lld, last_mcnt = %lld\n", (long long int)start_mcnt, (long long int)last_mcnt);
            #endif

            if (doDump) {
                xgpuClearDeviceIntegrationBuffer(&context);
                //xgpuReorderMatrix((Complex *)db_out->block[curblock_out].data);
                db_out->block[curblock_out].header.mcnt = start_mcnt;
                db_out->block[curblock_out].header.good_data = good_data;
                //printf("COR: Dumping correlator output with mcnt %lld\n", (long long int) start_mcnt);
            

                // Mark output block as full and advance
                #if VERBOSE==1
                printf("COR: Marking output block %d as filled, mcnt=%lld\n", curblock_out, (long long int)start_mcnt);
                #endif
                flag_frb_gpu_correlator_output_databuf_set_filled(db_out, curblock_out);
                curblock_out = (curblock_out + 1) % db_out->header.n_block;
                start_mcnt = last_mcnt + 1;
                last_mcnt = start_mcnt + int_count*N_MCNT_PER_FRB_BLOCK - 1;
                // Reset good_data flag for next block
                good_data = 1;
            }

            #if VERBOSE==1
            printf("COR: Marking input block %d as free\n", curblock_in);
            #endif
            flag_frb_gpu_input_databuf_set_free(db_in, curblock_in);
            curblock_in = (curblock_in + 1) % db_in->header.n_block;
        }
        }
        else if (cur_state == CLEANUP) {
            printf("COR: In Cleanup\n");
            next_state = ACQUIRE;
            // Set interntal integ_status to start
            hashpipe_status_lock_safe(&st);
            hputs(st.buf, "INTSTAT", "start");
            hashpipe_status_unlock_safe(&st);
            strcpy(integ_status, "start");

            // Clear out integration buffer on GPU
            xgpuClearDeviceIntegrationBuffer(&context);
            curblock_in = 0;
            curblock_out = 0;
            //start_mcnt = 0;
            //last_mcnt = 0;
            good_data = 1;
            hashpipe_status_lock_safe(&st);
            hputl(st.buf, "CLEANB", 1);
            hashpipe_status_unlock_safe(&st);
        }
        
        // Next state processing
        hashpipe_status_lock_safe(&st);
        switch (next_state) {
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
static hashpipe_thread_desc_t x_thread = {
    name: "flag_frb_correlator_thread",
    skey: "CORSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_frb_gpu_input_databuf_create},
    obuf_desc: {flag_frb_gpu_correlator_output_databuf_create}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&x_thread);
}

