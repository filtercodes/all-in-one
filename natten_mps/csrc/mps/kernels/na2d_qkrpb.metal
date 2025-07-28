#include <metal_stdlib>
#include "mps_kernel_structs.h"
#include "helpers.metal"

using namespace metal;

kernel void na2d_permute_and_reshape(
    device const float* context [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant const PermuteAndReshape2dProperties& props [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {

    const int x = gid.x;
    const int y = gid.y;
    const int b = gid.z;

    if (y >= props.height || x >= props.width) {
        return;
    }

    const int output_offset = b * props.output_stride_b + y * props.output_stride_y + x * props.output_stride_x;

    for (int h = 0; h < props.heads; ++h) {
        for (int d = 0; d < props.dim; ++d) {
            const int context_offset = b * props.context_stride_b + h * props.context_stride_h + y * props.context_stride_y + x * props.context_stride_x + d * props.context_stride_d;
            const int output_index = output_offset + (h * props.dim + d) * props.output_stride_d;
            output[output_index] = context[context_offset];
        }
    }
}

kernel void neighborhood_assembly_2d_qkrpb(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* rpb [[buffer(2)]],
    device float* attn [[buffer(3)]],
    constant const NA2dQkrpbProperties& props [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {

    const int x = gid.x;
    const int y = gid.y;
    const int z = gid.z;
    const int h = z % props.heads;
    const int b = z / props.heads;

    if (y >= props.height || x >= props.width) {
        return;
    }

    const int neighborhood_size = props.kernel_size / 2;
    const int query_offset = b * props.query_stride_b + h * props.query_stride_h + y * props.query_stride_y + x * props.query_stride_x;
    const int attn_offset = b * props.attn_stride_b + h * props.attn_stride_h + y * props.attn_stride_y + x * props.attn_stride_x;
    float attn_row[256];

    for (int ky = 0; ky < props.kernel_size; ++ky) {
        for (int kx = 0; kx < props.kernel_size; ++kx) {
            const int sample_y = get_window_start(y, props.height, props.kernel_size, neighborhood_size, props.dilation) + ky * props.dilation;
            const int sample_x = get_window_start(x, props.width, props.kernel_size, neighborhood_size, props.dilation) + kx * props.dilation;
            const int attn_index = ky * props.kernel_size + kx;

            if (sample_y >= 0 && sample_y < props.height && sample_x >= 0 && sample_x < props.width) {
                float sum = 0.0f;
                const int key_offset = b * props.key_stride_b + h * props.key_stride_h + sample_y * props.key_stride_y + sample_x * props.key_stride_x;
                const float query_scale = 1.0f / sqrt((float)props.dim);
                for (int d = 0; d < props.dim; ++d) {
                    sum += (query[query_offset + d * props.query_stride_d] * query_scale) * key[key_offset + d * props.key_stride_d];
                }

                const int rpb_y = get_pb_start(y, props.height, props.kernel_size, neighborhood_size, props.dilation);
                const int rpb_x = get_pb_start(x, props.width, props.kernel_size, neighborhood_size, props.dilation);
                const int rpb_offset = h * props.rpb_stride_h + (rpb_y + ky) * props.rpb_stride_y + (rpb_x + kx) * props.rpb_stride_x;
                attn_row[attn_index] = sum + rpb[rpb_offset];
            } else {
                attn_row[attn_index] = -1.0e+6f;
            }
        }
    }
    softmax(attn_row, props.kernel_size * props.kernel_size);

    for (int ky = 0; ky < props.kernel_size; ++ky) {
        for (int kx = 0; kx < props.kernel_size; ++kx) {
            const int attn_index = ky * props.kernel_size + kx;
            attn[attn_offset + attn_index * props.attn_stride_k] = attn_row[attn_index];
        }
    }
}

