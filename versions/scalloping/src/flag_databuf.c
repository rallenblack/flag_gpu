
#include "flag_databuf.h"

hashpipe_databuf_t * flag_input_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(flag_input_block_t);
    int    n_block     = N_INPUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int flag_input_databuf_wait_free(flag_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int flag_input_databuf_wait_filled(flag_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int flag_input_databuf_set_free(flag_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int flag_input_databuf_set_filled(flag_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

hashpipe_databuf_t * flag_gpu_input_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(flag_gpu_input_block_t);
    int    n_block     = N_GPU_INPUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int flag_gpu_input_databuf_wait_free(flag_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_input_databuf_wait_filled(flag_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_input_databuf_set_free(flag_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_input_databuf_set_filled(flag_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}



hashpipe_databuf_t * flag_frb_gpu_input_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(flag_frb_gpu_input_block_t);
    int    n_block     = N_GPU_FRB_INPUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int flag_frb_gpu_input_databuf_wait_free(flag_frb_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int flag_frb_gpu_input_databuf_wait_filled(flag_frb_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int flag_frb_gpu_input_databuf_set_free(flag_frb_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int flag_frb_gpu_input_databuf_set_filled(flag_frb_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

// Overloaded methods for the coarse correlator
hashpipe_databuf_t * flag_gpu_correlator_output_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(flag_gpu_correlator_output_block_t);
    int    n_block     = N_GPU_OUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int flag_gpu_correlator_output_databuf_wait_free(flag_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_correlator_output_databuf_wait_filled(flag_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_correlator_output_databuf_set_free(flag_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_correlator_output_databuf_set_filled(flag_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

//Overloaded methods for the frb correlator
hashpipe_databuf_t * flag_frb_gpu_correlator_output_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(flag_frb_gpu_correlator_output_block_t);
    int    n_block     = N_GPU_OUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int flag_frb_gpu_correlator_output_databuf_wait_free(flag_frb_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int flag_frb_gpu_correlator_output_databuf_wait_filled(flag_frb_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int flag_frb_gpu_correlator_output_databuf_set_free(flag_frb_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int flag_frb_gpu_correlator_output_databuf_set_filled(flag_frb_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

//Overloaded methods for the fine correlator
hashpipe_databuf_t * flag_pfb_gpu_correlator_output_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(flag_pfb_gpu_correlator_output_block_t);
    int    n_block     = N_GPU_OUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int flag_pfb_gpu_correlator_output_databuf_wait_free(flag_pfb_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int flag_pfb_gpu_correlator_output_databuf_wait_filled(flag_pfb_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int flag_pfb_gpu_correlator_output_databuf_set_free(flag_pfb_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int flag_pfb_gpu_correlator_output_databuf_set_filled(flag_pfb_gpu_correlator_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

// Overloaded methods for beamformer
hashpipe_databuf_t * flag_gpu_beamformer_output_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(flag_gpu_beamformer_output_block_t);
    int    n_block     = N_GPU_OUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int flag_gpu_beamformer_output_databuf_wait_free(flag_gpu_beamformer_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_beamformer_output_databuf_wait_filled(flag_gpu_beamformer_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_beamformer_output_databuf_set_free(flag_gpu_beamformer_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_beamformer_output_databuf_set_filled(flag_gpu_beamformer_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

//overloaded methods for pfb input buffers
hashpipe_databuf_t * flag_pfb_gpu_input_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(flag_pfb_gpu_input_block_t);
    int    n_block     = N_GPU_INPUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int flag_pfb_gpu_input_databuf_wait_free(flag_pfb_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int flag_pfb_gpu_input_databuf_wait_filled(flag_pfb_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int flag_pfb_gpu_input_databuf_set_free(flag_pfb_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int flag_pfb_gpu_input_databuf_set_filled(flag_pfb_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

void flag_pfb_gpu_input_databuf_clear(flag_pfb_gpu_input_databuf_t * d) {
    return hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

void flag_databuf_clear(hashpipe_databuf_t * d) {
    return hashpipe_databuf_clear(d);
}

// Overloaded methods for pfb output buffers
hashpipe_databuf_t * flag_gpu_pfb_output_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(flag_gpu_pfb_output_block_t);
    int    n_block     = N_GPU_OUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int flag_gpu_pfb_output_databuf_wait_free(flag_gpu_pfb_output_databuf_t* d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_pfb_output_databuf_wait_filled(flag_gpu_pfb_output_databuf_t* d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_pfb_output_databuf_set_free(flag_gpu_pfb_output_databuf_t* d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_pfb_output_databuf_set_filled(flag_gpu_pfb_output_databuf_t* d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

void flag_gpu_pfb_output_databuf_clear(flag_gpu_pfb_output_databuf_t* d) {
    return hashpipe_databuf_clear((hashpipe_databuf_t *)d);
}

//overloaded methods for total power
hashpipe_databuf_t * flag_gpu_power_output_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(flag_gpu_power_output_block_t);
    int    n_block     = N_GPU_OUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int flag_gpu_power_output_databuf_wait_free(flag_gpu_power_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_power_output_databuf_wait_filled(flag_gpu_power_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_power_output_databuf_set_free(flag_gpu_power_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int flag_gpu_power_output_databuf_set_filled(flag_gpu_power_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}



// Overloaded methods for databuf total status
int flag_pfb_gpu_correlator_output_databuf_total_status(flag_pfb_gpu_correlator_output_databuf_t * d) {
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

int flag_gpu_pfb_output_databuf_total_status(flag_gpu_pfb_output_databuf_t * d) {
    return hashpipe_databuf_total_status((hashpipe_databuf_t *)d);
}

int flag_pfb_gpu_input_databuf_total_status(flag_pfb_gpu_input_databuf_t * d) {
    return hashpipe_databuf_total_status((hashpipe_databuf_t *) d);
}

int flag_input_databuf_total_status(flag_input_databuf_t * d) {
    return hashpipe_databuf_total_status((hashpipe_databuf_t *) d);
}

int flag_gpu_correlator_output_databuf_total_status(flag_gpu_correlator_output_databuf_t * d) {
    return hashpipe_databuf_total_status((hashpipe_databuf_t *) d);
}
