/* flag_net_thread.c
 *
 * Routine to read packets from network and load them into the buffer.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <errno.h>

#include <xgpu.h>

#include "hashpipe.h"
#include "flag_databuf.h"

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#define ELAPSED_NS(start,stop) \
  (((int64_t)stop.tv_sec-start.tv_sec)*1000*1000*1000+(stop.tv_nsec-start.tv_nsec))


// Create thread status buffer
static hashpipe_status_t *st_p;


// Define a packet header type
// First 44 bits are the mcnt (system time index)
// Next 4 bits are the switching signal bits (BLANK | LO BLANK | CAL | SIG/REF)
// Next 8 bits are the Fengine ID from which the packet came
// Next 8 bits are the Xengine ID to which the packet is destined
typedef struct {
    uint64_t mcnt;
    uint8_t  cal;       // Noise Cal Status Mask
    int      fid;	// Fengine ID
    int      xid;	// Xengine ID
} packet_header_t;


// Define a block info type
// The output buffer takes blocks (collections of contiguous packets)
// The info structure keeps track of the following:
//     (1) The XID for this Xengine
//     (2) The reference starting mcnt
//         (the starting mcnt for the first block)
//     (3) The ID of the block that is currently being filled
// This structure will be static (i.e. it will reside in static memory, and not the stack)
typedef struct {
    int      initialized;                  // Boolean to indicate that block has been initialized
    int32_t  self_xid;                     // Xengine ID for this block
    uint64_t mcnt_start;                   // The starting mcnt for this block
    int      block_i;                      // The blocks ID
    int      packet_count[N_INPUT_BLOCKS]; // Packet counter for each block
    int      m;                            // Indices for packet payload destination
    int      f;                            //
} block_info_t;


// Method to initialize the block_info_t structure
static inline void initialize_block_info(block_info_t * binfo) {
    // If already initialized, return
    if (binfo->initialized) {
        return;
    }

    // Initialize our XID
    binfo->self_xid = -1;
    hashpipe_status_lock_safe(st_p);
    hgeti4(st_p->buf, "XID", &binfo->self_xid);
    hashpipe_status_unlock_safe(st_p);

    // Initialize packet counters
    int i;
    for (i = 0; i < N_INPUT_BLOCKS; i++) {
        binfo->packet_count[i] = 0;
    }

    // Initialize rest
    binfo->mcnt_start  = 0;
    binfo->block_i     = 0;
    binfo->initialized = 1;
}


// Method to compute the block index for a given packet mcnt
// Note that the mod operation allows the buffer to be circular
static inline int get_block_idx(uint64_t mcnt) {
    return (mcnt / Nm) % N_INPUT_BLOCKS;
}


// Method to print the header of a received packet
void print_pkt_header(packet_header_t * pkt_header) {

    static long long prior_mcnt;

    printf("packet header : mcnt %012lx (diff from prior %lld) cal %hx fid %d xid %d\n",
	   pkt_header->mcnt, pkt_header->mcnt-prior_mcnt, pkt_header->cal, pkt_header->fid, pkt_header->xid);

    prior_mcnt = pkt_header->mcnt;
}


// Method to extract a packet's header information and store it
static inline void get_header (struct hashpipe_udp_packet *p, packet_header_t * pkt_header) {
    uint64_t raw_header;
    raw_header = be64toh(*(unsigned long long *)p->data);
    // printf("raw_header: %016lx\n", raw_header);
    pkt_header->mcnt        = raw_header >> 20;
    pkt_header->cal         = (raw_header >> 16) & 0x000000000000000F;
    pkt_header->xid         = raw_header         & 0x00000000000000FF;
    pkt_header->fid         = (raw_header >> 8)  & 0x00000000000000FF;
}


// Method to calculate the buffer address for packet payload
// Also verifies FID and XID of packets
static inline int calc_block_indices(block_info_t * binfo, packet_header_t * pkt_header) {
    // Verify FID and XID
    if (pkt_header->fid >= Nf) {
        hashpipe_error(__FUNCTION__, "packet FID %u out of range (0-%d)", pkt_header->fid, Nf-1);
        return -1;
    }
    else if (pkt_header->xid != binfo->self_xid && binfo->self_xid != -1) {
        hashpipe_error(__FUNCTION__, "unexpected packet XID %d (expected %d)", pkt_header->xid, binfo->self_xid);
        return -1;
    }
    binfo->m = pkt_header->mcnt % Nm;
    binfo->f = pkt_header->fid;
    return 0; 
}


// Method to initialize a block's header information and establish
// the starting mcnt for the block
static inline void initialize_block(flag_input_databuf_t * db, uint64_t mcnt) {
    int block_idx = get_block_idx(mcnt);
    db->block[block_idx].header.good_data = 0;
    db->block[block_idx].header.mcnt_start = mcnt - (mcnt%Nm);
}


// Method to mark the block as filled
static void set_block_filled(flag_input_databuf_t * db, block_info_t * binfo) {
    
    static int last_filled = -1; // The last block that was filled

    uint32_t block_idx = get_block_idx(binfo->mcnt_start);
    
    // Validate that we're filling blocks in the proper sequence
    int next_filled = (last_filled + 1)% N_INPUT_BLOCKS;
    if (next_filled != block_idx) {
        hashpipe_warn(__FUNCTION__, "block %d being marked filled, but expected block %d!", block_idx, next_filled);
    }
    
    // Validate that block_idx matches binfo->block_i
    if (block_idx != binfo->block_i) {
        hashpipe_warn(__FUNCTION__, "block_idx (%d) != binfo->block_i (%d)", block_idx, binfo->block_i);
    }

    // Mark block as good if all packets are there
    if (binfo->packet_count[block_idx] == N_PACKETS_PER_BLOCK) {
        // RACE CONDITION !!!!!!!!!
        // We need to lock the buffer first before modifing it.
        // Arguably, this data will not be accessed by the next thread until the
        // block is marked as filled...
        db->block[block_idx].header.good_data = 1;
    }

    // Mark block as filled so next thread can process it
    last_filled = block_idx;
    flag_input_databuf_set_filled(db, block_idx);

    binfo->self_xid = -1;
    hashpipe_status_lock_safe(st_p);
    hgeti4(st_p->buf, "XID", &binfo->self_xid);
    hashpipe_status_unlock_safe(st_p);
}


// Method to process a received packet
// Processing involves the following
// (1) header extraction
// (2) block population (output buffer data type is a block)
// (3) buffer population (if block is filled)
static inline uint64_t process_packet(flag_input_databuf_t * db, struct hashpipe_udp_packet *p) {
    static block_info_t binfo;
    packet_header_t     pkt_header;

    // Initialize block information data types
    if (!binfo.initialized) {
        initialize_block_info(&binfo);
    }
    
    // Parse packet header
    get_header(p, &pkt_header);
    uint64_t pkt_mcnt  = pkt_header.mcnt;
    uint64_t cur_mcnt  = binfo.mcnt_start;
    int dest_block_idx = get_block_idx(pkt_mcnt);

    // Check mcnt to see if packet belongs in current block, next, or the one after
    int64_t pkt_mcnt_dist = pkt_mcnt - cur_mcnt;
   
    // If packet is for the current block + 2, then mark current block as full
    // and increment current block
    if (pkt_mcnt_dist >= 2*Nm && pkt_mcnt_dist < 3*Nm) { // 2nd next block (Current block + 2)
        set_block_filled(db, &binfo);

        // Advance mcnt_start to next block
        cur_mcnt += Nm;
        binfo.mcnt_start += Nm;
        binfo.block_i = (binfo.block_i + 1) % N_INPUT_BLOCKS;

        // Initialize next block
        flag_input_databuf_wait_free(db, dest_block_idx);
        initialize_block(db, pkt_mcnt);

        // Reset packet counter for this block
        binfo.packet_count[dest_block_idx] = 0;
    }
    else if (pkt_mcnt_dist >= 3*Nm) { // > current block + 2
        // The x-engine is lagging behind the f-engine, or the x-engine
        // has just started. Reinitialize the current block
        // to have the next multiple of Nm. Then initialize the next block appropriately
        uint64_t new_mcnt = pkt_mcnt - (pkt_mcnt % (Nm*N_INPUT_BLOCKS)) + Nm*N_INPUT_BLOCKS;
	// binfo.block_i = get_block_idx(new_mcnt);

        fprintf(stderr, "Packet mcnt %lld is very late... resettting current block mcnt to %lld (%012lx)\n", (long long int)pkt_mcnt, (long long int)new_mcnt, new_mcnt);
        fprintf(stderr, "pkt_mcnt_dist = %lld\n", (long long int)pkt_mcnt_dist);

        initialize_block(db, new_mcnt);
        binfo.packet_count[binfo.block_i] = 0;
        binfo.mcnt_start = new_mcnt;

        hashpipe_status_lock_safe(st_p);
        hputi4(st_p->buf, "NETMCNT", new_mcnt);
        hashpipe_status_unlock_safe(st_p);
        return -1;
    }
    else if (pkt_mcnt_dist < 0) {
        fprintf(stderr, "Early packet, pkt_mcnt_dist = %lld\n", (long long int)pkt_mcnt_dist);
        return -1;
    }
    // Increment packet count for block
    binfo.packet_count[dest_block_idx]++;

    // Validate FID and XID
    // Calculate "m" and "f" which index the buffer for writing packet payload
    if (calc_block_indices(&binfo, &pkt_header) == -1) {
        hashpipe_error(__FUNCTION__, "invalid FID and XID in header");
        return -1;
    }

    // Calculate starting points for writing packet payload into buffer
    // POSSIBLE RACE CONDITION!!!! Need to lock db->block access with semaphore
    uint64_t * dest_p  = db->block[dest_block_idx].data + flag_input_databuf_idx(binfo.m, binfo.f, 0, 0);
    const uint64_t * payload_p = (uint64_t *)(p->data+8); // Ignore header

    // Copy data into buffer
    memcpy(dest_p, payload_p, N_BYTES_PER_PACKET-8); // Ignore header

    //print_pkt_header(&pkt_header);

    return pkt_mcnt;
}


// Run method for the thread
// It is meant to do the following:
// (1) Initialize status buffer
// (2) Set up network parameters and socket
// (3) Start main loop
//     (3a) Receive packet on socket
//     (3b) Error check packet (packet size, etc)
//     (3c) Call process_packet on received packet
// (4) Terminate thread cleanly
static void *run(hashpipe_thread_args_t * args) {

    // Local aliases to shorten access to args fields
    // Our output buffer happens to be a paper_input_databuf
    flag_input_databuf_t *db = (flag_input_databuf_t *)args->obuf;
    hashpipe_status_t st = args->st;
    const char * status_key = args->thread_desc->skey;

    st_p = &st;	// allow global (this source file) access to the status buffer

    int tmp = -1;
    hashpipe_status_lock_safe(&st);
    hgeti4(st.buf, "XID", &tmp);
    hashpipe_status_unlock_safe(&st);


    /* Read network params */
    struct hashpipe_udp_params up = {
	.bindhost = "0.0.0.0",
	.bindport = 8511,
	.packet_size = N_BYTES_PER_PACKET
    };

    hashpipe_status_lock_safe(&st);
    	// Get info from status buffer if present (no change if not present)
    	hgets(st.buf, "BINDHOST", 80, up.bindhost);
    	hgeti4(st.buf, "BINDPORT", &up.bindport);
    
    	// Store bind host/port info etc in status buffer
    	hputs(st.buf, "BINDHOST", up.bindhost);
    	hputi4(st.buf, "BINDPORT", up.bindport);
    	hputu4(st.buf, "MISSEDFE", 0);
    	hputu4(st.buf, "MISSEDPK", 0);
    	hputs(st.buf, status_key, "running");
    hashpipe_status_unlock_safe(&st);

    struct hashpipe_udp_packet p;

    /* Give all the threads a chance to start before opening network socket */
    int netready = 0;
    int corready = 0;
    int checkready = 0;
    while (!netready) {
        sleep(1);
        // Check the correlator to see if it's ready yet
        hashpipe_status_lock_safe(&st);
        hgeti4(st.buf, "CORREADY",  &corready);
        hgeti4(st.buf, "SAVEREADY", &checkready);
        hashpipe_status_unlock_safe(&st);
        if (!corready) {
            continue;
        }
        //if (!checkready) {
        //    continue;
        //}

        // Check the other threads to see if they're ready yet
        // TBD

        // If we get here, then all threads are initialized
        netready = 1;
    }
    sleep(1);

    /* Set up UDP socket */
    fprintf(stderr, "NET: BINDHOST = %s\n", up.bindhost);
    fprintf(stderr, "NET: BINDPORT = %d\n", up.bindport);
    int rv = hashpipe_udp_init(&up);
    
    if (rv!=HASHPIPE_OK) {
        hashpipe_error("paper_net_thread",
                "Error opening UDP socket.");
        pthread_exit(NULL);
    }
    pthread_cleanup_push((void *)hashpipe_udp_close, &up);


    // Initialize first few blocks in the buffer
    int i;
    for (i = 0; i < 2; i++) {
        // Wait until block semaphore is free
        if (flag_input_databuf_wait_free(db, i) != HASHPIPE_OK) {
            if (errno == EINTR) { // Interrupt occurred
                hashpipe_error(__FUNCTION__, "waiting for free block interrupted\n");
                pthread_exit(NULL);
            } else {
                hashpipe_error(__FUNCTION__, "error waiting for free block\n");
                pthread_exit(NULL);
            }
        }
        initialize_block(db, i*Nm);
    }

    // Set correlator's starting mcnt to 0
    hashpipe_status_lock_safe(&st);
    hputi4(st.buf, "NETMCNT", 0);
    hashpipe_status_unlock_safe(&st);

    // Set correlator to "start" state
    hashpipe_status_lock_safe(&st);
    hputs(st.buf, "INTSTAT", "start");
    hashpipe_status_unlock_safe(&st);

    /* Main loop */
    uint64_t packet_count = 0;
    char integ_status[17];

    fprintf(stdout, "NET: Starting Thread!\n");
    
    while (run_threads()) {
        // Get packet
	do {
	    p.packet_size = recv(up.sock, p.data, HASHPIPE_MAX_PACKET_SIZE, 0);
	} while (p.packet_size == -1 && (errno == EAGAIN || errno == EWOULDBLOCK) && run_threads());
	if(!run_threads()) break;
       
        // Check packet size and report errors 
        if (up.packet_size != p.packet_size) {
	    // If an error was returned instead of a valid packet size
            if (p.packet_size == -1) {
                fprintf(stderr, "uh oh!\n");
		// Log error and exit
                hashpipe_error("paper_net_thread",
                        "hashpipe_udp_recv returned error");
                perror("hashpipe_udp_recv");
                pthread_exit(NULL);
            } else {
		// Log warning and ignore wrongly sized packet
                hashpipe_warn("paper_net_thread", "Incorrect pkt size (%d)", p.packet_size);
                continue;
            }
	}

	// Check to see if a stop command has been issued
	hashpipe_status_lock_safe(&st);
	hgets(st.buf, "INTSTAT", 16, integ_status);
	hashpipe_status_unlock_safe(&st);

	// Process packets only if we are not in a stop state
	if (strcmp(integ_status, "stop") != 0) {
            // Process packet
    	    packet_count++;
            process_packet(db, &p);    
	}

        /* Will exit if thread has been cancelled */
        pthread_testcancel();
    }

    pthread_cleanup_pop(1); /* Closes push(hashpipe_udp_close) */

    hashpipe_status_lock_busywait_safe(&st);
    hputs(st.buf, status_key, "terminated");
    hashpipe_status_unlock_safe(&st);
    return NULL;
}


static hashpipe_thread_desc_t net_thread = {
    name: "flag_net_thread",
    skey: "NETSTAT",
    init: NULL,
    run:  run,
    ibuf_desc: {NULL},
    obuf_desc: {flag_input_databuf_create}
};


static __attribute__((constructor)) void ctor() {
  register_hashpipe_thread(&net_thread);
}

// vi: set ts=8 sw=4 noet :
