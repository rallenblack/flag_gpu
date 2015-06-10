
#include "grating_databuf.h"

hashpipe_databuf_t * grating_input_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(grating_input_block_t);
    int    n_block     = N_INPUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int grating_input_databuf_wait_free(grating_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int grating_input_databuf_wait_filled(grating_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int grating_input_databuf_set_free(grating_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int grating_input_databuf_set_filled(grating_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}



hashpipe_databuf_t * grating_gpu_input_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(grating_gpu_input_block_t);
    int    n_block     = N_GPU_INPUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int grating_gpu_input_databuf_wait_free(grating_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int grating_gpu_input_databuf_wait_filled(grating_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int grating_gpu_input_databuf_set_free(grating_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int grating_gpu_input_databuf_set_filled(grating_gpu_input_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}



hashpipe_databuf_t * grating_output_databuf_create(int instance_id, int databuf_id) {
    size_t header_size = sizeof(hashpipe_databuf_t) + sizeof(hashpipe_databuf_cache_alignment);
    size_t block_size  = sizeof(grating_output_block_t);
    int    n_block     = N_COR_OUT_BLOCKS;
    return hashpipe_databuf_create(
        instance_id, databuf_id, header_size, block_size, n_block);
}

int grating_output_databuf_wait_free(grating_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_free((hashpipe_databuf_t *)d, block_id);
}

int grating_output_databuf_wait_filled(grating_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_wait_filled((hashpipe_databuf_t *)d, block_id);
}

int grating_output_databuf_set_free(grating_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_free((hashpipe_databuf_t *)d, block_id);
}

int grating_output_databuf_set_filled(grating_output_databuf_t * d, int block_id) {
    return hashpipe_databuf_set_filled((hashpipe_databuf_t *)d, block_id);
}

