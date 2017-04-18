/* flag_corcheck_thread.c
 *
 * Routine to save correlator outputs to file for data verification
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#include "hashpipe.h"
#include "flag_databuf.h"
#include <xgpu.h>

#define DISABLE_SAVE 1

typedef struct {
    flag_frb_gpu_correlator_output_databuf_t * db_in;
} x_save_args;

typedef struct {
    flag_gpu_beamformer_output_databuf_t * db_in;
} b_save_args;

// Create thread status buffer
static hashpipe_status_t * st_p;


void * run_corrsave_thread(void * args) {
    x_save_args * my_args = (x_save_args *)args;
    flag_frb_gpu_correlator_output_databuf_t * db_in = my_args->db_in;
    
    int rv;
    int curblock_in = 0;
    while (run_threads()) {
        
        // Wait for input buffer block to be filled
        while ((rv=flag_frb_gpu_correlator_output_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(st_p);
                hputs(st_p->buf, "CORRSAVE", "waiting for free block");
                hashpipe_status_unlock_safe(st_p);
            }
            else {
                hashpipe_error(__FUNCTION__, "error waiting for filled databuf block");
                pthread_exit(NULL);
                break;
            }
        }

        // Extract information from header
        uint64_t start_mcnt = db_in->block[curblock_in].header.mcnt;

        // Get filename for output correlation file
        char filename[256];
        sprintf(filename, "corr_mcnt_%lld.out", (long long)start_mcnt);
        if (start_mcnt % 200 == 0) {
            fprintf(stderr, "SAV: Saving to %s\n", filename);
        }

        #if DISABLE_SAVE == 0
        // Get pointer to input buffer
        Complex * p = (Complex *)db_in->block[curblock_in].data;
        
        // Open file and save
        FILE * filePtr = fopen(filename, "w");
        int j;
        for (j = 0; j < N_COR_MATRIX; j++) {
            float p_re = p[j].real;
            float p_im = p[j].imag;
            fprintf(filePtr, "%g\n", p_re);
            fprintf(filePtr, "%g\n", p_im);
        }
        fclose(filePtr);
        #endif

        // Mark input block as free and advance
        flag_frb_gpu_correlator_output_databuf_set_free(db_in, curblock_in);
        curblock_in = (curblock_in + 1) % db_in->header.n_block;
        pthread_testcancel();
    }

    return NULL;
}

void * run_beamsave_thread(void * args) {
    b_save_args * my_args = (b_save_args *)args;
    flag_gpu_beamformer_output_databuf_t * db_in = my_args->db_in;
    
    int rv;
    int curblock_in = 0;
    while (run_threads()) {
        
        // Wait for input buffer block to be filled
        while ((rv=flag_gpu_beamformer_output_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
            if (rv==HASHPIPE_TIMEOUT) {
                hashpipe_status_lock_safe(st_p);
                hputs(st_p->buf, "BEAMSAVE", "waiting for free block");
                hashpipe_status_unlock_safe(st_p);
            }
            else {
                hashpipe_error(__FUNCTION__, "error waiting for filled databuf block");
                pthread_exit(NULL);
                break;
            }
        }

        // Extract information from header
        uint64_t start_mcnt = db_in->block[curblock_in].header.mcnt;

        // Get filename for output correlation file
        char filename[256];
        sprintf(filename, "beam_mcnt_%lld.out", (long long)start_mcnt);
        if (start_mcnt % 200 == 0) {
            fprintf(stderr, "SAV: Saving to %s\n", filename);
        }

        #if DISABLE_SAVE == 0
        // Get pointer to input buffer
        float * p = (float *)db_in->block[curblock_in].data;
        
        // Open file and save
        FILE * filePtr = fopen(filename, "w");
        fwrite(p, sizeof(float), N_BEAM_SAMPS, filePtr);
        fclose(filePtr);
        #endif

        // Mark input block as free and advance
        flag_gpu_beamformer_output_databuf_set_free(db_in, curblock_in);
        curblock_in = (curblock_in + 1) % db_in->header.n_block;
        pthread_testcancel();
    }

    return NULL;
}


// Run method for the thread
static void * run(hashpipe_thread_args_t * args) {

    #if DISABLE_SAVE == 1
        printf("SAV: Saving feature disabled... no data will be saved!\n");
    #else
        printf("SAV: Slow saving feature enabled... high-speed performance will be limited!\n");
    #endif

    // Local aliases to shorten access to args fields
    flag_frb_gpu_correlator_output_databuf_t * db_in = (flag_frb_gpu_correlator_output_databuf_t  *)args->ibuf;
    hashpipe_status_t st = args->st;

    st_p = &st; // allow global (this source file) access to the status buffer

    // Create additional databuf just for the beamformer outputs
    flag_gpu_beamformer_output_databuf_t * bf_in = (flag_gpu_beamformer_output_databuf_t *) flag_gpu_beamformer_output_databuf_create(args->instance_id, args->input_buffer+2);

    // Set up the argument structures
    x_save_args x_args;
    x_args.db_in = db_in;

    b_save_args b_args;
    b_args.db_in = bf_in; 

    // Mark thread as ready to run
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "SAVREADY", 1);
    hashpipe_status_unlock_safe(&st);

    pthread_t corrsave_thread;
    pthread_t beamsave_thread;

    pthread_create(&corrsave_thread, NULL, run_corrsave_thread, &x_args);
    pthread_create(&beamsave_thread, NULL, run_beamsave_thread, &b_args);

    pthread_join(corrsave_thread, NULL);
    pthread_join(beamsave_thread, NULL);

    // Thread terminates after loop
    return NULL;
}

// Thread description
static hashpipe_thread_desc_t bxsave_thread = {
    name: "flag_bx_save_thread",
    skey: "DUALSAVE",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_frb_gpu_correlator_output_databuf_create},
    obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&bxsave_thread);
}

