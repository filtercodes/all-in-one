#include <metal_stdlib>
#include "mps_kernel_structs.h"
#include "helpers.metal"

using namespace metal;

kernel void na1d_permute_and_reshape(
    device const float* context [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant const PermuteAndReshape1dProperties& props [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {

    const int i = gid.x;
    const int b = gid.z;

    if (i >= props.length) {
        return;
    }

    const int output_offset = b * props.output_stride_b + i * props.output_stride_l;

    for (int h = 0; h < props.heads; ++h) {
        for (int d = 0; d < props.dim; ++d) {
            const int context_offset = b * props.context_stride_b + h * props.context_stride_h + i * props.context_stride_l + d * props.context_stride_d;
            const int output_index = output_offset + (h * props.dim + d) * props.output_stride_d;
            output[output_index] = context[context_offset];
        }
    }
}

kernel void neighborhood_assembly_1d_qkrpb(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* rpb [[buffer(2)]],
    device float* attn [[buffer(3)]],
    constant const NA1dQkrpbProperties& props [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {

    const int i = gid.x;
    const int h = gid.y;
    const int b = gid.z;

    if (i >= props.length) {
        return;
    }

    const int query_offset = b * props.query_stride_b + h * props.query_stride_h + i * props.query_stride_l;
    const int attn_offset = b * props.attn_stride_b + h * props.attn_stride_h + i * props.attn_stride_l;
    const int neighborhood_size = props.kernel_size / 2;

    for (int j = 0; j < props.kernel_size; ++j) {
        const int sample_i = get_window_start(i, props.length, props.kernel_size, neighborhood_size, props.dilation) + j * props.dilation;
        
        if (sample_i >= 0 && sample_i < props.length) {
            const int pi = get_pb_start(i, props.length, props.kernel_size, neighborhood_size, props.dilation);
            const int rpb_offset = h * props.rpb_stride_h + (pi + j) * props.rpb_stride_l;
            
            float sum = 0.0f;
            const int key_offset = b * props.key_stride_b + h * props.key_stride_h + sample_i * props.key_stride_l;
            const float query_scale = 1.0f / sqrt((float)props.dim);

            for (int d = 0; d < props.dim; ++d) {
                sum += (query[query_offset + d * props.query_stride_d] * query_scale) * key[key_offset + d * props.key_stride_d];
            }
            
            attn[attn_offset + j * props.attn_stride_k] = sum + rpb[rpb_offset];
        } else {
            attn[attn_offset + j * props.attn_stride_k] = 0.0f;
        }
    }
}



kernel void na1d_qkrpb_softmax(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* rpb [[buffer(2)]],
    device float* attn [[buffer(3)]],
    constant const NA1dQkrpbProperties& props [[buffer(4)]],
    device int* debug_sample_indices [[buffer(5)]],
    device int* debug_pb_indices [[buffer(6)]],
    device float* debug_query [[buffer(7)]],
    device float* debug_key [[buffer(8)]],
    device float* debug_rpb [[buffer(9)]],
    uint3 gid [[thread_position_in_grid]]) {

    const int i = gid.x;
    const int h = gid.y;
    const int b = gid.z;

    if (i >= props.length) {
        return;
    }

    const int query_offset = b * props.query_stride_b + h * props.query_stride_h + i * props.query_stride_l;
    const int attn_offset = b * props.attn_stride_b + h * props.attn_stride_h + i * props.attn_stride_l;
    const int neighborhood_size = props.kernel_size / 2;

    float attn_row[256];

    for (int j = 0; j < props.kernel_size; ++j) {
        const int sample_i = get_window_start(i, props.length, props.kernel_size, neighborhood_size, props.dilation) + j * props.dilation;
        
        if (sample_i >= 0 && sample_i < props.length) {
            const int pi = get_pb_start(i, props.length, props.kernel_size, neighborhood_size, props.dilation);
            const int rpb_offset = h * props.rpb_stride_h + (pi + j) * props.rpb_stride_l;
            
            float sum = 0.0f;
            const int key_offset = b * props.key_stride_b + h * props.key_stride_h + sample_i * props.key_stride_l;
            const float query_scale = 1.0f / sqrt((float)props.dim);

            for (int d = 0; d < props.dim; ++d) {
                sum += (query[query_offset + d * props.query_stride_d] * query_scale) * key[key_offset + d * props.key_stride_d];
            }
            
            attn_row[j] = sum + rpb[rpb_offset];
        } else {
            attn_row[j] = -1.0e+6f;
        }
    }

    softmax(attn_row, props.kernel_size);

    for (int j = 0; j < props.kernel_size; ++j) {
        attn[attn_offset + j * props.attn_stride_k] = attn_row[j];
    }
}