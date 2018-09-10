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

// Create thread status buffer
static hashpipe_status_t * st_p;

// Run method for the thread
static void * run(hashpipe_thread_args_t * args) {

    // Local aliases to shorten access to args fields
    flag_gpu_correlator_output_databuf_t * db_in = (flag_gpu_correlator_output_databuf_t  *)args->ibuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    st_p = &st; // allow global (this source file) access to the status buffer

    // Mark thread as ready to run
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "SAVREADY", 1);
    hashpipe_status_unlock_safe(&st);

    int rv;
    int curblock_in = 0;
    while (run_threads()) {
        
        // Wait for input buffer block to be filled
        while ((rv=flag_gpu_correlator_output_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
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

        char directory[128];
        char BANK[5];
        hashpipe_status_lock_safe(&st);
        hgets(st.buf, "DATADIR", 127, directory);
        hgets(st.buf, "BANKNAM", 5, BANK);
        hashpipe_status_unlock_safe(&st);

       

        uint64_t start_mcnt = db_in->block[curblock_in].header.mcnt;
        //int64_t good_data = db_in->block[curblock_in].header.good_data;
        char filename[256];
        sprintf(filename, "%s/TGBT16A_508_01/TMP/BF/cor_mcnt_%lld_%s.out", directory, (long long)start_mcnt, BANK);
        fprintf(stderr, "SAV: Saving to %s\n", filename);
        
        #if SAVE == 1
            Complex * p = (Complex *)db_in->block[curblock_in].data;
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
        

        flag_gpu_correlator_output_databuf_set_free(db_in, curblock_in);
        curblock_in = (curblock_in + 1) % db_in->header.n_block;
        pthread_testcancel();
    }

    // Thread terminates after loop
    return NULL;
}

// Thread description
static hashpipe_thread_desc_t xsave_thread = {
    name: "flag_corsave_thread",
    skey: "CORSAVE",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_gpu_correlator_output_databuf_create},
    obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&xsave_thread);
}
