/* flag_powersave_thread.c
 *
 * Routine to save total power outputs to file for data verification
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>

#include "hashpipe.h"
#include "flag_databuf.h"
#include <xgpu.h>
#include "pfb.h"

// Create thread status buffer
static hashpipe_status_t * st_p;

// Run method for the thread
static void * run(hashpipe_thread_args_t * args) {

    // Local aliases to shorten access to args fields
    flag_gpu_pfb_output_databuf_t * db_in = (flag_gpu_pfb_output_databuf_t  *)args->ibuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    st_p = &st; // allow global (this source file) access to the status buffer

    int instance_id = args[0].instance_id;
    char data_dir[128];
    hashpipe_status_lock_safe(&st);
    hgets(st.buf, "DATADIR", 127, data_dir);
    hashpipe_status_unlock_safe(&st);
    if (data_dir == NULL) {
        printf("SAV: DATADIR = .\n");
    }
    else {
        printf("SAV: DATADIR = %s\n", data_dir);
    }

    // Mark thread as ready to run
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "SAVREADY", 1);
    hashpipe_status_unlock_safe(&st);

    int rv;
    int curblock_in = 0;
    while (run_threads()) {
        
        // Wait for input buffer block to be filled
        while ((rv=flag_gpu_pfb_output_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
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
        printf("Getting ready to save...\n");

        uint64_t start_mcnt = db_in->block[curblock_in].header.mcnt;
        int good_data_flag = (int)(db_in->block[curblock_in].header.good_data);
        float * pfb_out_data = (float *)db_in->block[curblock_in].data;

	printf("Creating filename...\n");
        char filename[256];
        sprintf(filename, "%s/pfb_%d_mcnt_%lld.out", data_dir, instance_id, (long long)start_mcnt);
	printf("Saving output file to: %s\n", filename);
        FILE * filePtr = fopen(filename, "wb");
	if(filePtr == NULL) {
		printf("DUM DUM...\n");
	}
	printf("Writing good data...\n");
        fwrite(&good_data_flag, sizeof(int), 1, filePtr);
	printf("Writing data out...\n");
        fwrite(pfb_out_data, sizeof(float), PFB_OUTPUT_BLOCK_SIZE, filePtr);
        fclose(filePtr);

        flag_gpu_pfb_output_databuf_set_free(db_in, curblock_in);
        curblock_in = (curblock_in + 1) % db_in->header.n_block;
        pthread_testcancel();
    }

    // Thread terminates after loop
    return NULL;
}

// Thread description
static hashpipe_thread_desc_t pfbsave_thread = {
    name: "flag_pfbsave_thread",
    skey: "PFBSAVE",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_gpu_pfb_output_databuf_create},
    obuf_desc: {NULL}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&pfbsave_thread);
}
