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
    flag_gpu_input_databuf_t * db_in = (flag_gpu_input_databuf_t *)args->ibuf;
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

    state cur_state = ACQUIRE;
    state next_state = ACQUIRE;
    int64_t good_data = 1;
    char weight_flag[8];
    char netstat[17];
    char weight_file[17];

    // Indicate in shared memory buffer that this thread is ready to start
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "CORREADY", 1);
    hashpipe_status_unlock_safe(&st);
    
    int check_count = 0;
    // Main loop for thread
    while (run_threads()) {
        
        if(cur_state == ACQUIRE){
            next_state = ACQUIRE;
	        // Wait for input buffer block to be filled
            while ((rv=flag_gpu_input_databuf_wait_filled(db_in, curblock_in)) != HASHPIPE_OK) {
                if (rv==HASHPIPE_TIMEOUT) {
                    int cleanb;
                    hashpipe_status_lock_safe(&st);
                    hgetl(st.buf, "CLEANB", &cleanb);
                    hgets(st.buf, "NETSTAT", 16, netstat);
                    hgets(st.buf, "WFLAG", 8, weight_flag);
                    hashpipe_status_unlock_safe(&st);
                    if (cleanb == 0 && strcmp(netstat, "CLEANUP") == 0) {
                        next_state = CLEANUP;
                        printf("BF: Entering CLEANUP state\n");
                        break;
                    }
                    if (strcmp(weight_flag,"1") == 0){
                        hashpipe_status_lock_safe(&st);
                        hgets(st.buf,"BWEIFILE",16,weight_file);
                        hashpipe_status_unlock_safe(&st);

                        sprintf(w_dir, "%s\%s", weightdir, weight_file);
                        printf("BF: Weight file name: %s\n", w_dir);
                        

                        printf("RTB: Initializing beamformer weights...\n");
                        // update_weights(weight_file);
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
                        
                       hashpipe_status_lock_safe(&st);
                       hputs(st.buf,"WFLAG","0");
                       hashpipe_status_unlock_safe(&st); 
                    }
                }
                else {
                    hashpipe_error(__FUNCTION__, "error waiting for filled databuf block");
                    pthread_exit(NULL);
                    break;
                }
            }

            // If CLEANUP, don't continue processing
            if (next_state != CLEANUP) {
                

		    // Print out the header information for this block 
		    flag_gpu_input_header_t tmp_header;
		    memcpy(&tmp_header, &db_in->block[curblock_in].header, sizeof(flag_gpu_input_header_t));
		    good_data = tmp_header.good_data;

		    hashpipe_status_lock_safe(&st);
		    hputi4(st.buf, "BEAMMCNT", tmp_header.mcnt);
		    hashpipe_status_unlock_safe(&st);

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
		    check_count++;
		   // if(check_count == 1000){
		   // }
			// Get block's starting mcnt for output block
		    db_out->block[curblock_out].header.mcnt = tmp_header.mcnt;
		    db_out->block[curblock_out].header.good_data = good_data;
            //printf("BF: good_data = %lld\n", (long long int)good_data);
		        
		    // Mark output block as full and advance
		    #if VERBOSE==1
		        printf("BF: Setting block %d, mcnt %lld as filled\n", curblock_out, (long long int)tmp_header.mcnt);
		    #endif
		    flag_gpu_beamformer_output_databuf_set_filled(db_out, curblock_out);
		    curblock_out = (curblock_out + 1) % db_out->header.n_block;
		    start_mcnt = last_mcnt + 1;
		    last_mcnt = start_mcnt + Nm - 1;
		    
		    // Mark input block as free
		    flag_gpu_input_databuf_set_free(db_in, curblock_in);
		    curblock_in = (curblock_in + 1) % db_in->header.n_block;
		}
	}
	else if (cur_state == CLEANUP) {
	    next_state = ACQUIRE;
	    curblock_in = 0;
	    curblock_out = 0;
	    hashpipe_status_lock_safe(&st);
	    hputl(st.buf, "CLEANB",1);
	    hashpipe_status_unlock_safe(&st);
	    printf("BF: Finished CLEANUP, returning to ACQUIRE\n");
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
    return NULL;
}



// Thread description
static hashpipe_thread_desc_t b_thread = {
    name: "flag_beamform_thread",
    skey: "BEAMSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {flag_gpu_input_databuf_create},
    obuf_desc: {flag_gpu_beamformer_output_databuf_create}
};

static __attribute__((constructor)) void ctor() {
    register_hashpipe_thread(&b_thread);
}

