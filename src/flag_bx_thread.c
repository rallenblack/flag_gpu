/* flag_bx_thread.c
 * 
 * Routine to run the fast-dump reduced bandwidth correlator and beamformer
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
    flag_frb_gpu_input_databuf_t * db_in;
    flag_frb_gpu_correlator_output_databuf_t * db_out;
    char * integ_status;
    int good_data;
    int int_count;
    uint64_t start_mcnt;
    uint64_t last_mcnt;
    XGPUContext context;
    state next_state;
    state cur_state;
    char * status_key;
} x_args;

// Arguments for the beamformer thread
typedef struct {
    int curblock_in;
    int curblock_out;
    flag_gpu_input_databuf_t * db_in;
    flag_gpu_beamformer_output_databuf_t * db_out;
    state next_state;
    state cur_state;
    char * status_key;
    int good_data;
    uint64_t start_mcnt;
    uint64_t last_mcnt;
} b_args;

// Thread function for correlator
void * run_correlator_thread(void * args) {

    x_args * my_args = (x_args *)args;

    // Create local access to status shared memory
    hashpipe_status_t st = *(st_p);

    // Extract arguments into local memory
    flag_frb_gpu_input_databuf_t * db_in = my_args->db_in;
    flag_frb_gpu_correlator_output_databuf_t * db_out = my_args->db_out;

    // Other variables
    char netstat[17];
    int  rv;

    while (run_threads()) {
        // ---------------------------------- ACQUIRE STATE ----------------------------------
        if (my_args->cur_state == ACQUIRE) {
            my_args->next_state = ACQUIRE;
    	    // Wait for input buffer block to be filled
            while ((rv=flag_frb_gpu_input_databuf_wait_filled(db_in, my_args->curblock_in)) != HASHPIPE_OK) {
                // If we timeout, check if NET thread is in CLEANUP state
                if (rv==HASHPIPE_TIMEOUT) {
                    int cleanb;
                    char msg[60];
                    sprintf(msg, "waiting for input block %d", my_args->curblock_in);
                    hashpipe_status_lock_safe(&st);
                    hgetl(st.buf, "CLEANB", &cleanb);
                    hgets(st.buf, "NETSTAT", 16, netstat);
                    hputs(st.buf, "CORBUFF", msg);
                    hashpipe_status_unlock_safe(&st);
                    // If NET in CLEANUP state, switch to CLEANUP and stop waiting
                    if (cleanb == 0 && strcmp(netstat, "CLEANUP") == 0) {
                       printf("COR: Cleanup condition met!\n");
                       my_args->next_state = CLEANUP;
                       break;
                    }
                }
                else {
                    hashpipe_error(__FUNCTION__, "error waiting for filled databuf block");
                    pthread_exit(NULL);
                    break;
                }
            }

            if (my_args->next_state != CLEANUP) {
                // Print out the header information for this block 
                flag_gpu_input_header_t tmp_header;
                memcpy(&tmp_header, &db_in->block[my_args->curblock_in].header, sizeof(flag_gpu_input_header_t));
	            //printf("COR: Received block %d, starting mcnt = %lld\n", curblock_in, (long long int)tmp_header.mcnt);
                my_args->good_data &= tmp_header.good_data;

                hashpipe_status_lock_safe(&st);
                hputi4(st.buf, "CORMCNT", tmp_header.mcnt);
                hashpipe_status_unlock_safe(&st);

                // Retrieve correlator integrator status
                hashpipe_status_lock_safe(&st);
                hgets(st.buf, "INTSTAT", 16, my_args->integ_status);
                hashpipe_status_unlock_safe(&st);
            
                // If the correlator integrator status is "off,"
                // Free the input block and continue
                if (strcmp(my_args->integ_status, "off") == 0) {
                    #if VERBOSE==1
                        fprintf(stderr, "COR: Correlator is off...\n");
                    #endif
                    flag_frb_gpu_input_databuf_set_free(db_in, my_args->curblock_in);
                    my_args->curblock_in = (my_args->curblock_in + 1) % db_in->header.n_block;
                    my_args->good_data = 1;
                    continue;
                }

                // If the correlator integrator status is "start,"
                // Get the correlator started
                // The INTSTAT string is set to "start" by the net thread once it's up and running
                if (strcmp(my_args->integ_status, "start") == 0) {

	            // Get the starting mcnt for integration (should be zero)
                    hashpipe_status_lock_safe(&st);
                    hgeti4(st.buf, "NETMCNT", (int *)(&(my_args->start_mcnt)));
                    hashpipe_status_unlock_safe(&st); 

                    // Check to see if block's starting mcnt matches INTSYNC
                    if (db_in->block[my_args->curblock_in].header.mcnt < my_args->start_mcnt) {

		                // If we get here, then there is a bug since the net thread shouldn't
                        // mark blocks as filled that are before the starting mcnt
                        // fprintf(stderr, "COR: Unable to start yet... waiting for mcnt = %lld\n", (long long int)start_mcnt);

                        // starting mcnt not yet reached -> free block and continue
                        flag_frb_gpu_input_databuf_set_free(db_in, my_args->curblock_in);
                        my_args->curblock_in = (my_args->curblock_in + 1) % db_in->header.n_block;
                        continue;
                    }
                    else if (db_in->block[my_args->curblock_in].header.mcnt == my_args->start_mcnt) {
                        // set correlator integrator to "on"
                        strcpy(my_args->integ_status, "on");
                        float requested_integration_time = 0.0;
                        float actual_integration_time = 0.0;
                        hashpipe_status_lock_safe(&st);
                        hputs(st.buf, "INTSTAT", my_args->integ_status);
                        hgetr4(st.buf, "REQSTI", &requested_integration_time);
                        hashpipe_status_unlock_safe(&st);

                        // Calculate integration times
                        my_args->int_count = ceil((N_MCNT_PER_SECOND / N_MCNT_PER_FRB_BLOCK) * requested_integration_time);
                        actual_integration_time = my_args->int_count/(N_MCNT_PER_SECOND / N_MCNT_PER_FRB_BLOCK);

                        hashpipe_status_lock_safe(&st);
                        hputr4(st.buf, "ACTSTI", actual_integration_time);
                        hputi4(st.buf, "INTCOUNT", my_args->int_count);
                        hashpipe_status_unlock_safe(&st);

                        // Compute last mcount
                        my_args->last_mcnt = my_args->start_mcnt + my_args->int_count*N_MCNT_PER_FRB_BLOCK - 1;
                    }
                    else {
                        // fprintf(stdout, "COR: We missed the start of the integration\n");
	            	    fprintf(stdout, "COR: Missed start. Expected start_mcnt = %lld, got %lld\n", (long long int)my_args->start_mcnt, (long long int)db_in->block[my_args->curblock_in].header.mcnt);
                        // we apparently missed the start of the integation... ouch...
                    }
                }

	            // Check to see if a stop is issued
        	    if (strcmp(my_args->integ_status, "stop") == 0) {
	                continue;
	            }

                // If we get here, then integ_status == "on"
                // Setup for current chunk
                my_args->context.input_offset  = my_args->curblock_in  * sizeof(flag_frb_gpu_input_block_t) / sizeof(ComplexInput);
                my_args->context.output_offset = my_args->curblock_out * sizeof(flag_frb_gpu_correlator_output_block_t) / sizeof(Complex);
            
                int doDump = 0;
                if ((db_in->block[my_args->curblock_in].header.mcnt + my_args->int_count*N_MCNT_PER_FRB_BLOCK - 1) >= my_args->last_mcnt) {
                    doDump = 1;

                    // Wait for new output block to be free
                    while ((rv=flag_frb_gpu_correlator_output_databuf_wait_free(db_out, my_args->curblock_out)) != HASHPIPE_OK) {
                        if (rv==HASHPIPE_TIMEOUT) {
                            char msg[60];
                            sprintf(msg, "waiting for output block %d", my_args->curblock_out);
                            int cleanb;
                            hashpipe_status_lock_safe(&st);
                            hgetl(st.buf, "CLEANB", &cleanb);
                            hgets(st.buf, "NETSTAT", 16, netstat);
                            hputs(st.buf, "CORBUFF", msg);
                            hashpipe_status_unlock_safe(&st);
                            if (cleanb == 0 && strcmp(netstat, "CLEANUP") == 0) {
                               printf("COR: Cleanup condition met!\n");
                               my_args->next_state = CLEANUP;
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
           
                xgpuCudaXengine(&(my_args->context), doDump ? SYNCOP_DUMP : SYNCOP_SYNC_TRANSFER);
           
                #if VERBOSE==1
                printf("COR: start_mcnt = %lld, last_mcnt = %lld\n", (long long int)(my_args->start_mcnt), (long long int)(my_args->last_mcnt));
                #endif
                if (doDump) {
                    xgpuClearDeviceIntegrationBuffer(&(my_args->context));
                    //xgpuReorderMatrix((Complex *)db_out->block[curblock_out].data);
                    db_out->block[my_args->curblock_out].header.mcnt = my_args->start_mcnt;
                    db_out->block[my_args->curblock_out].header.good_data = my_args->good_data;
                    //printf("COR: Dumping correlator output with mcnt %lld\n", (long long int) start_mcnt);
                

                    // Mark output block as full and advance
                    #if VERBOSE==1
                    printf("COR: Marking output block %d as filled, mcnt=%lld\n", my_args->curblock_out, (long long int)(my_args->start_mcnt));
                    #endif
                    flag_frb_gpu_correlator_output_databuf_set_filled(db_out, my_args->curblock_out);
                    my_args->curblock_out = (my_args->curblock_out + 1) % db_out->header.n_block;
                    my_args->start_mcnt = my_args->last_mcnt + 1;
                    my_args->last_mcnt = my_args->start_mcnt + my_args->int_count*N_MCNT_PER_FRB_BLOCK -1;
                    // Reset good_data flag for next block
                    my_args->good_data = 1;
                }

                #if VERBOSE==1
                printf("COR: Marking input block %d as free\n", my_args->curblock_in);
                #endif
                flag_frb_gpu_input_databuf_set_free(db_in, my_args->curblock_in);
                my_args->curblock_in = (my_args->curblock_in + 1) % db_in->header.n_block;
                #if VERBOSE==1
                printf("COR: Will now look for input block %d, mcnt=%lld\n", my_args->curblock_in, (long long int)(my_args->start_mcnt));
                #endif
            }
        }
        // ---------------------------------- CLEANUP STATE ----------------------------------
        else if (my_args->cur_state == CLEANUP) {
            // Simply kill ourselves and return
            return NULL;
        }
        // ---------------------------------- ERROR STATE ------------------------------------
        else if (my_args->cur_state == ERROR) {
            return NULL;
        }
        
        // Next state processing
        hashpipe_status_lock_safe(&st);
        switch (my_args->next_state) {
            case ACQUIRE: hputs(st.buf, my_args->status_key, "ACQUIRE"); break;
            case CLEANUP: hputs(st.buf, my_args->status_key, "CLEANUP"); break;
            case ERROR:   hputs(st.buf, my_args->status_key, "ERROR"); break;
        }
        hashpipe_status_unlock_safe(&st);
        my_args->cur_state = my_args->next_state;
        pthread_testcancel();
    }

    // Thread terminates after loop
    return NULL;
}

// Thead function for beamformer
void  * run_beamformer_thread(void * args) {

    b_args * my_args = (b_args *)args;

    // Create local access to status shared memory
    hashpipe_status_t st = *(st_p);

    // Extract arguments into local memory
    flag_gpu_input_databuf_t * db_in = my_args->db_in;
    flag_gpu_beamformer_output_databuf_t * db_out = my_args->db_out;
    
    // Propagate good_data flag
    my_args->start_mcnt = db_in->block[my_args->curblock_in].header.mcnt;
    my_args->good_data = db_in->block[my_args->curblock_in].header.good_data;

    // Other variables
    char netstat[17];
    int  rv;
    int  check_count = 0;

    // Begin main loop
    while (run_threads()) {
        
        // ---------------------------------- ACQUIRE STATE ----------------------------------
        if(my_args->cur_state == ACQUIRE){
            my_args->next_state = ACQUIRE;
	        // Wait for input buffer block to be filled
            while ((rv=flag_gpu_input_databuf_wait_filled(db_in, my_args->curblock_in)) != HASHPIPE_OK) {
                // If we timeout, check to see if the NET thread is in a CLEANUP state
                if (rv==HASHPIPE_TIMEOUT) {
                    int cleanb;
                    char msg[60];
                    sprintf(msg, "waiting for input block %d", my_args->curblock_in);
                    hashpipe_status_lock_safe(&st);
                    hgetl(st.buf, "CLEANB", &cleanb);
                    hgets(st.buf, "NETSTAT", 16, netstat);
                    hputs(st.buf, "BEAMBUFF", msg);
                    hashpipe_status_unlock_safe(&st);
                    // If NET thread is in CLEANUP state, change to CLEANUP state and stop waiting
                    if (cleanb == 0 && strcmp(netstat, "CLEANUP") == 0) {
                        my_args->next_state = CLEANUP;
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
            memcpy(&tmp_header, &db_in->block[my_args->curblock_in].header, sizeof(flag_gpu_input_header_t));
            my_args->good_data = tmp_header.good_data;
            hashpipe_status_lock_safe(&st);
            hputi4(st.buf, "BEAMMCNT", tmp_header.mcnt);
            hashpipe_status_unlock_safe(&st);

            #if VERBOSE == 1
                printf("BF: Received input block %d, starting mcnt %lld \n", my_args->curblock_in, (long long int)(tmp_header.mcnt));
            #endif

            // Wait for output block to become free
            while ((rv=flag_gpu_beamformer_output_databuf_wait_free(db_out, my_args->curblock_out)) != HASHPIPE_OK) {
                // If we timeout, just try again
                if (rv==HASHPIPE_TIMEOUT) {
                    char msg[60];
                    sprintf(msg, "waiting for output block %d", my_args->curblock_out);
                    hashpipe_status_lock_safe(&st);
                    hputs(st.buf, "BEAMBUFF", msg);
                    hashpipe_status_unlock_safe(&st);
                    continue;
                } else {
                    hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                    fprintf(stderr, "rv = %d\n", rv);
                    pthread_exit(NULL);
                    break;
                }
            }
           
            // Run the beamformer
            #if VERBOSE == 1
            printf("BF: Starting Beamformer!\n");
            #endif
            run_beamformer((signed char *)&db_in->block[my_args->curblock_in].data, (float *)&db_out->block[my_args->curblock_out].data);
            check_count++;
            // if(check_count == 1000){
            #if VERBOSE == 1
                 printf("RTBF: dumping mcnt = %lld\n", (long long int)(my_args->start_mcnt));
            #endif
            // }
	        // Get block's starting mcnt for output block
            db_out->block[my_args->curblock_out].header.mcnt = my_args->start_mcnt;
            db_out->block[my_args->curblock_out].header.good_data = my_args->good_data;
                
            #if VERBOSE == 1
                printf("BF: Marking output block %d as filled, starting mcnt %lld \n", my_args->curblock_out, (long long int)(tmp_header.mcnt));
            #endif
            // Mark output block as full and advance
            flag_gpu_beamformer_output_databuf_set_filled(db_out, my_args->curblock_out);
            my_args->curblock_out = (my_args->curblock_out + 1) % db_out->header.n_block;
            my_args->start_mcnt = my_args->last_mcnt + 1;
            my_args->last_mcnt = my_args->start_mcnt + Nm - 1;
            
            #if VERBOSE == 1
                printf("BF: Marking input block %d as free\n", my_args->curblock_in);
            #endif
            // Mark input block as free
            flag_gpu_input_databuf_set_free(db_in, my_args->curblock_in);
            my_args->curblock_in = (my_args->curblock_in + 1) % db_in->header.n_block;
        }
        // ---------------------------------- CLEANUP STATE ----------------------------------
        else if (my_args->cur_state == CLEANUP) {
            // If we are in a CLEANUP state, simply return
            return NULL;
        }
        // ---------------------------------- ERROR STATE ------------------------------------
        else if (my_args->cur_state == ERROR) {
            return NULL;
        }

        // Next state processing
        hashpipe_status_lock_safe(&st);
        switch(my_args->next_state){
            case ACQUIRE: hputs(st.buf, my_args->status_key, "ACQUIRE"); break;
            case CLEANUP: hputs(st.buf, my_args->status_key, "CLEANUP"); break;
            case ERROR:   hputs(st.buf, my_args->status_key, "ERROR"); break;
        }
        hashpipe_status_unlock_safe(&st);
        my_args->cur_state = my_args->next_state;
        pthread_testcancel();
    }

    // Thread terminates after loop
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
    flag_frb_gpu_input_databuf_t * db_in = (flag_frb_gpu_input_databuf_t *)args->ibuf;
    flag_frb_gpu_correlator_output_databuf_t * db_out = (flag_frb_gpu_correlator_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    // int instance_id = args->instance_id;

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



    /***************************************************************************
     * BEAMFORMER INIT
     ***************************************************************************/

    // Need to create another databuf just for the beamformer inputs
    printf("Creating databuf %d for beamformer inputs\n", args->input_buffer+2);
    flag_gpu_input_databuf_t * bf_in = (flag_gpu_input_databuf_t *)flag_gpu_input_databuf_create(args->instance_id, args->input_buffer+2);

    // Need to create another databuf just for the beamformer outputs
    printf("Creating databuf %d for beamformer outputs\n", args->output_buffer+2);
    flag_gpu_beamformer_output_databuf_t * bf_out = (flag_gpu_beamformer_output_databuf_t *)flag_gpu_beamformer_output_databuf_create(args->instance_id, args->output_buffer+2);

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
    int      curblock_in     = 0;
    int      curblock_out    = 0;
    int      curblock_bf_in  = 0;
    int      curblock_bf_out = 0;
    uint64_t start_mcnt      = 0;
    uint64_t last_mcnt       = 0;
    int      int_count       = 1;
    int64_t  good_data       = 1;

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
    my_x_args.cur_state    = ACQUIRE;
    my_x_args.status_key   = "CORSTAT";

    b_args my_b_args;
    my_b_args.curblock_in  = curblock_bf_in;
    my_b_args.curblock_out = curblock_bf_out;
    my_b_args.db_in        = bf_in;
    my_b_args.db_out       = bf_out;
    my_b_args.good_data    = good_data;
    my_b_args.next_state   = ACQUIRE;
    my_b_args.cur_state    = ACQUIRE;
    my_b_args.start_mcnt   = start_mcnt;
    my_b_args.last_mcnt    = last_mcnt;
    my_b_args.status_key   = "BEAMSTAT";

    // Create the correlator thread
    pthread_t corr_thread;
    // Only run for even-indexed instances
    if (args->instance_id % 2 == 0) {
        pthread_create(&corr_thread, NULL, run_correlator_thread, &my_x_args);
    }
    #if VERBOSE == 1
    else {
        printf("BX: Only running beamformer for this instance!\n");
    }
    #endif

    // Create the beamformer thread
    pthread_t beam_thread;
    pthread_create(&beam_thread, NULL, run_beamformer_thread, &my_b_args);

    /***************************************************************************
     * Begin Loop
     ***************************************************************************/

    // Mark thread as ready to run
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "CORREADY", 1);
    hashpipe_status_unlock_safe(&st);
 
    while (run_threads()) {

        if (args->instance_id % 2 == 0) {
            pthread_join(corr_thread, NULL);
            printf("BX: Correlator thread has joined\n");
        }
        pthread_join(beam_thread, NULL);
        printf("BX: Beamformer thread has joined\n");

        
        // TODO: Check current state of each thread to see if CLEANUP state
        if (my_b_args.cur_state == CLEANUP) {
            // Reinitialize counters
            curblock_in     = 0;
            curblock_out    = 0;
            curblock_bf_in  = 0;
            curblock_bf_out = 0;
            start_mcnt      = 0;
            last_mcnt       = 0;
            int_count       = 1;
            good_data       = 1;

            // Write to the correlator arguments
            my_x_args.curblock_in  = curblock_in;
            my_x_args.curblock_out = curblock_out;
            my_x_args.good_data    = good_data;
            my_x_args.int_count    = int_count;
            my_x_args.start_mcnt   = start_mcnt;
            my_x_args.last_mcnt    = last_mcnt;
            my_x_args.next_state   = ACQUIRE;
            my_x_args.cur_state    = ACQUIRE;
            my_x_args.status_key   = "CORSTAT";

            // Write to the beamformer arguments
            my_b_args.curblock_in  = curblock_bf_in;
            my_b_args.curblock_out = curblock_bf_out;
            my_b_args.good_data    = good_data;
            my_b_args.next_state   = ACQUIRE;
            my_b_args.cur_state    = ACQUIRE;
            my_b_args.start_mcnt   = start_mcnt;
            my_b_args.last_mcnt    = last_mcnt;
            my_b_args.status_key   = "BEAMSTAT";

            // Recreate threads
            printf("BX: Launching threads again\n");
            if (args->instance_id % 2 == 0) {
                pthread_create(&corr_thread, NULL, run_correlator_thread, &my_x_args);
            }
            pthread_create(&beam_thread, NULL, run_beamformer_thread, &my_b_args);
        }
        else if (my_x_args.cur_state == ERROR || my_b_args.cur_state == ERROR) {
            printf("BX: ERROR state reached. Terminating!\n");
            break;
        }
        else {
            printf("BX: Terminating both threads.\n");
        }

        pthread_testcancel();
    }

    // Thread terminates after loop
    return NULL;
}


// Thread description
static hashpipe_thread_desc_t bx_thread = {
    name: "flag_bx_thread",
    skey: "BXSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_frb_gpu_input_databuf_create},
    obuf_desc: {flag_frb_gpu_correlator_output_databuf_create}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&bx_thread);
}

