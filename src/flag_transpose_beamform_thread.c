/* flag_beamform_thread.c
 * 
 * Routine to form beams from received packets
 *  
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <sys/time.h>

#include "cublas_beamformer.h"
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
    flag_input_databuf_t * db_in = (flag_input_databuf_t *)args->ibuf;
    flag_gpu_beamformer_output_databuf_t * db_out = (flag_gpu_beamformer_output_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    st_p = &st; // allow global (this source file) access to the status buffer

    int rv;
    int curblock_in = 0;
    int curblock_out = 0;
    uint64_t start_mcnt = 0;
    uint64_t last_mcnt = Nm - 1;

    // Initialize beamformer
    printf("RTB: Initializing the beamformer...\n");
    init_beamformer();

    // Update weights
    // TODO: allow update of weights during runtime
    printf("RTB: Initializing beamformer weights...\n");
    char weightdir[65];
    hashpipe_status_lock_safe(&st);
    hgets(st.buf,"WEIGHTD", 65, weightdir);
    hashpipe_status_unlock_safe(&st);
    
    char w_dir[70];
    sprintf(w_dir, "%s/weights.in", weightdir);
    printf("BF: Weight file name: %s\n", w_dir);

    // update_weights("./weights.in");
    update_weights(w_dir);
    // Put metadata into status shared memory
    float offsets[BN_BEAM];
    char cal_filename[65];
    char algorithm[65];
    char weight_filename[65];
    long long unsigned int bf_xid;
    int act_xid;

    bf_get_offsets(offsets);
    bf_get_cal_filename(cal_filename);
    bf_get_algorithm(algorithm);
    bf_get_weight_filename(weight_filename);
    bf_xid = bf_get_xid();

    int i;
    hashpipe_status_lock_safe(&st);
    for (i = 0; i < BN_BEAM/2; i++) {
        char keyword1[9];
        snprintf(keyword1,8,"ELOFF%d",i);
        hputr4(st.buf, keyword1, offsets[2*i]);
        char keyword2[9];
        snprintf(keyword2,8,"AZOFF%d",i);
        hputr4(st.buf, keyword2, offsets[2*i+1]);
    }
    hputs(st.buf, "BCALFILE", cal_filename);
    hputs(st.buf, "BALGORIT", algorithm);
    hputs(st.buf, "BWFILE", weight_filename);
    hgeti4(st.buf, "XID", &act_xid);
    hashpipe_status_unlock_safe(&st);
    printf("RTBF: Finished weight initialization at beginning of thread...\n");

    state cur_state = ACQUIRE;
    state next_state = ACQUIRE;

    int weight_flag;
    char netstat[17];
    char weight_file[17];

    // Indicate in shared memory buffer that this thread is ready to start
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "RBFREADY", 1);
    hashpipe_status_unlock_safe(&st);
    
    // Main loop for thread
    while (run_threads()) {
        
        if(cur_state == ACQUIRE){
            next_state = ACQUIRE;
	        // Wait for input buffer block to be filled
            while ((rv=flag_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK && run_threads()) {
                if (rv==HASHPIPE_TIMEOUT) {
                    int cleanA;
                    hashpipe_status_lock_safe(&st);
                    hgetl(st.buf, "CLEANA", &cleanA);
                    hgets(st.buf, "NETSTAT", 16, netstat);
                    hgetl(st.buf, "WFLAG", &weight_flag);
                    hashpipe_status_unlock_safe(&st);
                    if (cleanA == 0 && strcmp(netstat, "CLEANUP") == 0) {
                        next_state = CLEANUP;
                        break;
                    }
                    if(weight_flag) {
                        hashpipe_status_lock_safe(&st);
                        hgets(st.buf,"BWEIFILE",16,weight_file);
                        hashpipe_status_unlock_safe(&st);

                        printf("RTBF: Starting the update weights proccess.\n");
                        sprintf(w_dir, "%s/%s", weightdir, weight_file);
                        printf("RTBF: Weight file name: %s\n", w_dir);
                        
                        printf("RTBF: Initializing beamformer weights...\n");
                        update_weights(w_dir);
                        printf("RTBF: Finished call to update_weights...\n");
                        // Put metadata into status shared memory
                        float offsets[BN_BEAM];
                        char cal_filename[65];
                        char algorithm[65];
                        char weight_filename[65];
                        long long unsigned int bf_xid;
                        int act_xid;

                        printf("RTBF: setting offsets...\n");
                        bf_get_offsets(offsets);
                        printf("RTBF: getting cal filename...\n");
                        bf_get_cal_filename(cal_filename);
                        printf("RTBF: getting algorithm...\n");
                        bf_get_algorithm(algorithm);
                        bf_get_weight_filename(weight_filename);
                        printf("RTBF: getting weight filename...\n");
                        bf_xid = bf_get_xid();
                     
                        int i;
                        printf("RTBF: updating Az/El Offset metadata...\n");
                        hashpipe_status_lock_safe(&st);
                        for (i = 0; i < BN_BEAM/2; i++) {
                            char keyword1[9];
                            snprintf(keyword1,8,"ELOFF%d",i);
                            hputr4(st.buf, keyword1, offsets[2*i]);
                            char keyword2[9];
                            snprintf(keyword2,8,"AZOFF%d",i);
                            hputr4(st.buf, keyword2, offsets[2*i+1]);
                        }
                        hputs(st.buf, "BCALFILE", cal_filename);
                        hputs(st.buf, "BALGORIT", algorithm);
                        hputs(st.buf, "BWFILE", weight_filename);
                        hgeti4(st.buf, "XID", &act_xid);
                        hputl(st.buf,"WFLAG",0);
                        hashpipe_status_unlock_safe(&st);
                        printf("RTBF: Finished weight update proccess.\n");
                    }
                }
                else {
                    hashpipe_error(__FUNCTION__, "error waiting for filled databuf block");
                    pthread_exit(NULL);
                    break;
                }
            }
            if (!run_threads()) break;

            // If CLEANUP, don't continue processing
            if (next_state != CLEANUP) {

                if (DEBUG) {
                    // Print out the header information for this block 
                    flag_input_header_t tmp_header;
                    memcpy(&tmp_header, &db_in->block[curblock_in].header, sizeof(flag_input_header_t));
                    hashpipe_status_lock_safe(&st);
                    hputi4(st.buf, "BEAMMCNT", tmp_header.mcnt_start);
                    hashpipe_status_unlock_safe(&st);
                }

                // Wait for output block to become free
                while ((rv=flag_gpu_beamformer_output_databuf_wait_free(db_out, curblock_out)) != HASHPIPE_OK) {
                    if (rv==HASHPIPE_TIMEOUT) {
                        continue;
                    } else {
                        hashpipe_error(__FUNCTION__, "error waiting for free databuf");
                        fprintf(stderr, "rv = %d\n", rv);
                        pthread_exit(NULL);
                        break;
                    }
                }
		   
                // Run the beamformer
                struct timeval tval_before, tval_after, tval_result;
                gettimeofday(&tval_before, NULL);
                run_beamformer((signed char *)&db_in->block[curblock_in].data, (float *)&db_out->block[curblock_out].data);
                gettimeofday(&tval_after, NULL);
                timersub(&tval_after, &tval_before, &tval_result);
                if ((float) tval_result.tv_usec/1000 > 13) {
                    printf("RTBF: Warning!!!!!!!!! Time = %f ms\n", (float) tval_result.tv_usec/1000);
                }
           
                // Get block's starting mcnt for output block
                db_out->block[curblock_out].header.mcnt = db_in->block[curblock_in].header.mcnt_start;
                db_out->block[curblock_out].header.good_data = db_in->block[curblock_in].header.good_data;		        

                if (VERBOSE) {
                    printf("BF: Setting block %d, mcnt %lld as filled\n", curblock_out, (long long int)db_out->block[curblock_out].header.mcnt);
                }
                // Mark output block as full and advance
                flag_gpu_beamformer_output_databuf_set_filled(db_out, curblock_out);
                curblock_out = (curblock_out + 1) % db_out->header.n_block;
                start_mcnt = last_mcnt + 1;
                last_mcnt = start_mcnt + Nm - 1;
		    
                // Mark input block as free
                flag_input_databuf_set_free(db_in, curblock_in);
                curblock_in = (curblock_in + 1) % db_in->header.n_block;
            }

        } else if (cur_state == CLEANUP) {
            if (VERBOSE) {
                printf("RTBF: In Cleanup\n");
            }

            hashpipe_status_lock_safe(&st);
            hgets(st.buf, "NETSTAT", 16, netstat);
            hashpipe_status_unlock_safe(&st);

            if (strcmp(netstat, "IDLE") == 0) {
                next_state = ACQUIRE;
                flag_databuf_clear((hashpipe_databuf_t *) db_out);
                printf("RTBF: Finished CLEANUP, clearing output databuf and returning to ACQUIRE\n");
            } else {
                next_state = CLEANUP;
                curblock_in = 0;
                curblock_out = 0;
                hashpipe_status_lock_safe(&st);
                hputl(st.buf, "CLEANA",1);
                hashpipe_status_unlock_safe(&st);
            }
        }

        // Next state processing
        hashpipe_status_lock_safe(&st);
        switch(next_state){
            case ACQUIRE: hputs(st.buf, status_key, "ACQUIRE"); break;
            case CLEANUP: hputs(st.buf, status_key, "CLEANUP"); break;        
        }
        hashpipe_status_unlock_safe(&st);
        cur_state = next_state;
        pthread_testcancel();
    }

    // Thread terminates after loop
    hashpipe_status_lock_busywait_safe(&st);
    printf("RTBF: Cleaning up gpu context...\n");
    rtbfCleanup();
    hputs(st.buf, status_key, "terminated");
    hashpipe_status_unlock_safe(&st);
    return NULL;
}



// Thread description
static hashpipe_thread_desc_t b_thread = {
    name: "flag_transpose_beamform_thread",
    skey: "BEAMSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_input_databuf_create},
    obuf_desc: {flag_gpu_beamformer_output_databuf_create}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&b_thread);
}

