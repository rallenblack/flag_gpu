/* flag_correlator_thread.c
 * 
 * Routine to correlate received packets
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>

#include "cublas_beamformer.h"
#include <xgpu.h>
#include "hashpipe.h"
#include "flag_databuf.h"

// Thread status buffer
static hashpipe_status_t * st_p;

// Enumerated types for flag_transpose_thread state machine
typedef enum {
    ACQUIRE,
    CLEANUP,
    ERROR
} state;

// Arguments for the correlator thread
typedef struct {
    int curblock_in;
    int curblock_out;
    flag_gpu_input_databuf_t * db_in;
    flag_gpu_correlator_output_databuf_t * db_out;
    char * integ_status;
    int good_data;
    int int_count;
    uint64_t start_mcnt;
    uint64_t last_mcnt;
    XGPUContext context;
    state next_state;
} x_args;

// Arguments for the beamformer thread
typedef struct {
    int curblock_in;
    int curblock_out;
    flag_gpu_input_databuf_t * db_in;
    flag_gpu_beamformer_output_databuf_t * db_out;
    state next_state;
} b_args;

// Thread function for correlator
void * run_correlator_thread(void * args) {

    x_args * my_args = (x_args *)args;

    // Extract arguments into local memory
    char * integ_status = my_args->integ_status;
    flag_gpu_input_databuf_t * db_in = my_args->db_in;
    flag_gpu_correlator_output_databuf_t * db_out = my_args->db_out;
    
    // Propagate good_data flag
    (my_args->good_data) &= db_in->block[my_args->curblock_in].header.good_data;

    // Retrieve correlator integrator status
    hashpipe_status_lock_safe(st_p);
    hgets(st_p->buf, "INTSTAT", 16, integ_status);
    hashpipe_status_unlock_safe(st_p);
        
    // If the correlator integrator status is "off,"
    // Return and reset variables
    if (strcmp(integ_status, "off") == 0) {
        // The input databuf semaphore control is reserved to the parent thread
        // flag_gpu_input_databuf_set_free(db_in, curblock_in);
        // args->curblock_in = (curblock_in + 1) % db_in->header.n_block;
        my_args->good_data = 1;
        return NULL;
    }
    
    // If the correlator integrator status is "start,"
    // Get the correlator started
    // The INTSTAT string is set to "start" by the net thread once it's up and running
    if (strcmp(integ_status, "start") == 0) {

        // Get the starting mcnt for integration (should be zero)
        hashpipe_status_lock_safe(st_p);
        hgeti4(st_p->buf, "NETMCNT", (int *)(&(my_args->start_mcnt)));
        hashpipe_status_unlock_safe(st_p);

        // Check to see if block's starting mcnt matches INTSYNC
        if (db_in->block[my_args->curblock_in].header.mcnt < my_args->start_mcnt) {

	    // If we get here, then there is a bug since the net thread shouldn't
	    // mark blocks as filled that are before the starting mcnt

            // starting mcnt not yet reached
            // free block and continue
            //flag_gpu_input_databuf_set_free(db_in, curblock_in);
            //curblock_in = (curblock_in + 1) % db_in->header.n_block;
            return NULL;
        }
        else if (db_in->block[my_args->curblock_in].header.mcnt == my_args->start_mcnt) {
            // set correlator integrator to "on"
            // fprintf(stderr, "COR: Starting correlator!\n");
            strcpy(integ_status, "on");
            float requested_integration_time = 0.0;
            float actual_integration_time = 0.0;
            hashpipe_status_lock_safe(st_p);
            hputs(st_p->buf, "INTSTAT", integ_status);
            hgetr4(st_p->buf, "REQSTI", &requested_integration_time);
            hashpipe_status_unlock_safe(st_p);

            my_args->int_count = ceil((N_MCNT_PER_SECOND / Nm) * requested_integration_time);
            actual_integration_time = (my_args->int_count)/(N_MCNT_PER_SECOND / Nm);

            hashpipe_status_lock_safe(st_p);
            hputr4(st_p->buf, "ACTSTI", actual_integration_time);
            hputi4(st_p->buf, "INTCOUNT", my_args->int_count);
            hashpipe_status_unlock_safe(st_p);

            // Compute last mcount
            my_args->last_mcnt = my_args->start_mcnt + (my_args->int_count)*Nm - 1;
        }
        else {
            // fprintf(stdout, "COR: We missed the start of the integration\n");
	    fprintf(stdout, "COR: Missed start. Expected start_mcnt = %lld, got %lld\n", (long long int)(my_args->start_mcnt), (long long int)db_in->block[my_args->curblock_in].header.mcnt);
            // we apparently missed the start of the integation... ouch...
        }
    }

    // Check to see if a stop is issued
    if (strcmp(integ_status, "stop") == 0) {
        return NULL;
    }

    // If we get here, then integ_status == "on"
    // Setup for current chunk
    my_args->context.input_offset  = my_args->curblock_in  * sizeof(flag_gpu_input_block_t) / sizeof(ComplexInput);
    my_args->context.output_offset = my_args->curblock_out * sizeof(flag_gpu_correlator_output_block_t) / sizeof(Complex);
    
    // Check to see if the correlator will dump to the output buffer    
    int doDump = 0;
    if ((db_in->block[my_args->curblock_in].header.mcnt + (my_args->int_count)*Nm - 1) >= my_args->last_mcnt) {
        doDump = 1;

        // Wait for new output block to be free
        int rv;
        while ((rv=flag_gpu_correlator_output_databuf_wait_free(db_out, my_args->curblock_out)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) { // If timeout, check if processing is done
                int cleanb;
                char netstat[17];
                hashpipe_status_lock_safe(st_p);
                hgetl(st_p->buf, "CLEANB", &cleanb);
                hgets(st_p->buf, "NETSTAT", 16, netstat);
                hashpipe_status_unlock_safe(st_p);
                if (cleanb == 0 && strcmp(netstat, "CLEANUP") == 0) {
                    printf("COR: Cleanup condition met!\n");
                    my_args->next_state = CLEANUP;
                    return NULL;
                }
            }
            else {
                hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                my_args->next_state = ERROR;
                return NULL;
            }
        }
    }
    
    // Run the correlator kernel  
    xgpuCudaXengine(&(my_args->context), doDump ? SYNCOP_DUMP : SYNCOP_SYNC_TRANSFER);
        
    // If the correlator dumped, clean up for next integration
    if (doDump) {
        xgpuClearDeviceIntegrationBuffer(&(my_args->context));
        db_out->block[my_args->curblock_out].header.mcnt = my_args->start_mcnt;
        db_out->block[my_args->curblock_out].header.good_data = my_args->good_data;
        
        // Mark output block as full and advance
        flag_gpu_correlator_output_databuf_set_filled(db_out, my_args->curblock_out);
        my_args->curblock_out = (my_args->curblock_out + 1) % db_out->header.n_block;
        my_args->start_mcnt = my_args->last_mcnt + 1;
        my_args->last_mcnt  = my_args->start_mcnt + (my_args->int_count)*Nm - 1;
        
        // Reset good_data flag for next block
        my_args->good_data = 1;
    }

    // Input buffer operations are reserved for master thread
    // flag_gpu_input_databuf_set_free(db_in, curblock_in);
    // curblock_in = (curblock_in + 1) % db_in->header.n_block;
    return NULL;
}

// Thead function for beamformer
void  * run_beamformer_thread(void * args) {

    b_args * my_args = (b_args *)args;

    // Extract arguments into local memory
    flag_gpu_input_databuf_t * db_in = my_args->db_in;
    flag_gpu_beamformer_output_databuf_t * db_out = my_args->db_out;
    
    // Propagate good_data flag
    uint64_t start_mcnt = db_in->block[my_args->curblock_in].header.mcnt;
    int good_data = db_in->block[my_args->curblock_in].header.good_data;
    
    // Wait for an output block to become free
    int rv;
    while ((rv=flag_gpu_beamformer_output_databuf_wait_free(db_out, my_args->curblock_out)) != HASHPIPE_OK) {
        if (rv == HASHPIPE_TIMEOUT) {
            int cleanb;
            char netstat[17];
            hashpipe_status_lock_safe(st_p);
            hgetl(st_p->buf, "CLEANB", &cleanb);
            hgets(st_p->buf, "NETSTAT", 16, netstat);
            hashpipe_status_unlock_safe(st_p);
            if (cleanb == 0 && strcmp(netstat, "CLEANUP") == 0) {
                printf("RTB: Cleanup condition met!\n");
                my_args->next_state = CLEANUP;
                return NULL;
            }
        }
        else {
            hashpipe_error(__FUNCTION__, "error waiting for free databuf");
            my_args->next_state = ERROR;
            return NULL;
        }
    }

    // Run the beamformer kernel
    run_beamformer((signed char *)&db_in->block[my_args->curblock_in].data, (float *)&db_out->block[my_args->curblock_out].data);

    // Update the output buffer header
    db_out->block[my_args->curblock_out].header.mcnt      = start_mcnt;
    db_out->block[my_args->curblock_out].header.good_data = good_data;

    // Mark output block as full and advance
    flag_gpu_beamformer_output_databuf_set_filled(db_out, my_args->curblock_out);
    my_args->curblock_out = (my_args->curblock_out + 1) % db_out->header.n_block;

    return NULL;
}


// Run method for the thread
// It is meant to do the following:
//     (1) Initialize status buffer
//     (2) Start main loop
//         (2a) Wait for input buffer block to be filled
//         (2b) Print out some data in the block
static void * run(hashpipe_thread_args_t * args) {
    // Local aliases to shorten access to args fields
    flag_gpu_input_databuf_t * db_in = (flag_gpu_input_databuf_t *)args->ibuf;
    flag_gpu_correlator_output_databuf_t * db_out = (flag_gpu_correlator_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;
    int instance_id = args->instance_id;
    printf("instance_id = %d\n", instance_id);

    st_p = &st; // allow global (this source file) access to the status buffer


    /***************************************************************************
     * CORRELATOR INIT
     ***************************************************************************/

    // Initialize correlator integrator status to "off"
    // Initialize starting mcnt to 0 (INTSYNC)
    char integ_status[17];
    int gpu_dev = 0;
    hashpipe_status_lock_safe(&st);
    hputs(st.buf, "INTSTAT", "off");
    hputi8(st.buf, "INTSYNC", 0);
    hputr4(st.buf, "REQSTI", 0.5); // Requested STI length (set by Dealer/Player)
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
   
    context.array_len = (db_in->header.n_block * sizeof(flag_gpu_input_block_t) - sizeof(flag_gpu_input_header_t))/sizeof(ComplexInput);
    context.matrix_len = (db_out->header.n_block * sizeof(flag_gpu_correlator_output_block_t) - sizeof(flag_gpu_output_header_t))/sizeof(Complex);

    int xgpu_error = xgpuInit(&context, gpu_dev);
    if (XGPU_OK != xgpu_error) {
        fprintf(stderr, "ERROR: xGPU initialization failed (error code %d)\n", xgpu_error);
        return THREAD_ERROR;
    }



    /***************************************************************************
     * BEAMFORMER INIT
     ***************************************************************************/

    // Need to create another databuf just for the beamformer outputs
    printf("Creating databuf %d for beamformer outputs\n", args->output_buffer+1);
    flag_gpu_beamformer_output_databuf_t * bf_out = (flag_gpu_beamformer_output_databuf_t *)flag_gpu_beamformer_output_databuf_create(args->instance_id, args->output_buffer+1);

    // Call beamformer_lib initialization functions
    init_beamformer();
    update_weights("./weights.in");

    // Allocate variables for metadata
    float offsets[BN_BEAM];
    char cal_filename[65];
    char algorithm[65];
    char weight_filename[65];
    long long unsigned int bf_xid;
    int act_xid;

    // Retrieve metadata from beamformer_lib
    bf_get_offsets(offsets);
    bf_get_cal_filename(cal_filename);
    bf_get_algorithm(algorithm);
    bf_get_weight_filename(weight_filename);
    bf_xid = bf_get_xid();

    // Add metadata to shared memory
    int i;
    hashpipe_status_lock_safe(&st);
    for (i = 0; i < BN_BEAM/2; i++) {
        char keyword1[9];
        snprintf(keyword1, 8, "ELOFF%d", i);
        hputr4(st.buf, keyword1, offsets[2*i]);
        char keyword2[9];
        snprintf(keyword2, 8, "AZOFF%d", i);
        hputr4(st.buf, keyword2, offsets[2*i+1]);
    }
    hputs(st.buf, "BCALFILE", cal_filename);
    hputs(st.buf, "BALGORIT", algorithm);
    hputs(st.buf, "BWEIFILE", weight_filename);
    hgeti4(st.buf, "XID", &act_xid);
    hashpipe_status_unlock_safe(&st);

    printf("RTB: Weight Filename = %s\n", weight_filename);


    /***************************************************************************
     * Initialize argument structures for threads
     ***************************************************************************/
    int      rv;
    char     netstat[17];
    int      curblock_in     = 0;
    int      curblock_out    = 0;
    int      curblock_bf_out = 0;
    uint64_t start_mcnt      = 0;
    uint64_t last_mcnt       = 0;
    int      int_count       = 1;
    int64_t  good_data       = 1;
    state    cur_state       = ACQUIRE;
    state    next_state      = ACQUIRE;

    x_args my_x_args;
    my_x_args.curblock_in  = curblock_in;
    my_x_args.curblock_out = curblock_out;
    my_x_args.db_in        = db_in;
    my_x_args.db_out       = db_out;
    my_x_args.integ_status = integ_status;
    my_x_args.good_data    = good_data;
    my_x_args.int_count    = int_count;
    my_x_args.start_mcnt   = start_mcnt;
    my_x_args.last_mcnt    = last_mcnt;
    my_x_args.context      = context;
    my_x_args.next_state   = ACQUIRE;

    b_args my_b_args;
    my_b_args.curblock_in  = curblock_in;
    my_b_args.curblock_out = curblock_bf_out;
    my_b_args.db_in        = db_in;
    my_b_args.db_out       = bf_out;
    my_b_args.next_state   = ACQUIRE;

    /***************************************************************************
     * Begin Loop
     ***************************************************************************/

    // Mark thread as ready to run
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

           // Prepare arguments for next block
           my_x_args.curblock_in = curblock_in;
           my_x_args.next_state = ACQUIRE;
           my_b_args.curblock_in = curblock_in;
           my_b_args.next_state = ACQUIRE;

           // Create the correlator thread
           pthread_t corr_thread;
           pthread_t beam_thread;
           pthread_create(&corr_thread, NULL, run_correlator_thread, &my_x_args);
           pthread_create(&beam_thread, NULL, run_beamformer_thread, &my_b_args);

           // Wait until all threads have joined
           pthread_join(corr_thread, NULL);
           pthread_join(beam_thread, NULL);

           // Update input buffer counters
           flag_gpu_input_databuf_set_free(db_in, curblock_in);
           curblock_in = (curblock_in + 1) % db_in->header.n_block;

           // Check if correlator/beamformer detected a clean-up condition
           if (my_x_args.next_state == CLEANUP || my_b_args.next_state == CLEANUP) {
               next_state = CLEANUP;
           }

           // Check if correlator/beamformer detected an error state
           if (my_x_args.next_state == ERROR || my_b_args.next_state == ERROR) {
               pthread_exit(NULL);
               break;
           }
           
       }
       else if (cur_state == CLEANUP) {
           printf("COR: In Cleanup\n");
           next_state = ACQUIRE;
           // Set interntal integ_status to start
           strcpy(integ_status, "start");

           // Clear out integration buffer on GPU
           xgpuClearDeviceIntegrationBuffer(&context);
           curblock_in = 0;
           curblock_out = 0;
           good_data = 1;
           hashpipe_status_lock_safe(&st);
           hputl(st.buf, "CLEANB", 1);
           hashpipe_status_unlock_safe(&st);
       }
       else if (cur_state == ERROR) {
           pthread_exit(NULL);
           break;
       }

        // Next state processing
        hashpipe_status_lock_safe(&st);
        switch (next_state) {
            case ACQUIRE: hputs(st.buf, status_key, "ACQUIRE"); break;
            case CLEANUP: hputs(st.buf, status_key, "CLEANUP"); break;
            case ERROR:   hputs(st.buf, status_key, "ERROR"); break;
        }
        hashpipe_status_unlock_safe(&st);
        cur_state = next_state;
        pthread_testcancel();
    }

    // Thread terminates after loop
    return NULL;
}


// Thread description
static hashpipe_thread_desc_t d_thread = {
    name: "flag_dual_thread",
    skey: "DUALSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_gpu_input_databuf_create},
    obuf_desc: {flag_gpu_correlator_output_databuf_create}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&d_thread);
}

