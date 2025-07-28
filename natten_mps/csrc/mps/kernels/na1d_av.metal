#include <metal_stdlib>
#include "mps_kernel_structs.h"
#include "helpers.metal"

using namespace metal;

kernel void neighborhood_assembly_1d_av(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant const NA1dAVProperties& props [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
    
    const int i = gid.x;
    const int h = gid.y;
    const int b = gid.z;

    if (i >= props.length) {
        return;
    }

    const int neighborhood_size = props.kernel_size / 2;
    const int attn_offset = b * props.attn_stride_b + h * props.attn_stride_h + i * props.attn_stride_l;
    const int output_offset = b * props.output_stride_b + h * props.output_stride_h + i * props.output_stride_l;

    for (int d = 0; d < props.dim; ++d) {
        float sum = 0.0f;
        for (int j = 0; j < props.kernel_size; ++j) {
            const int sample_i = get_window_start(i, props.length, props.kernel_size, neighborhood_size, props.dilation) + j * props.dilation;
            
            if (sample_i >= 0 && sample_i < props.length) {
                const int value_offset = b * props.value_stride_b + h * props.value_stride_h + sample_i * props.value_stride_l;
                sum += attn[attn_offset + j * props.attn_stride_k] * value[value_offset + d * props.value_stride_d];
            }
        }
        output[output_offset + d * props.output_stride_d] = sum;
    }
}
