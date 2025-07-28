#include <metal_stdlib>
#include "mps_kernel_structs.h"
#include "helpers.metal"

using namespace metal;

kernel void neighborhood_assembly_2d_av(
    device const float* attn [[buffer(0)]],
    device const float* value [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant const NA2dAVProperties& props [[buffer(3)]],
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
    const int attn_offset = b * props.attn_stride_b + h * props.attn_stride_h + y * props.attn_stride_y + x * props.attn_stride_x;
    const int output_offset = b * props.output_stride_b + h * props.output_stride_h + y * props.output_stride_y + x * props.output_stride_x;

    for (int d = 0; d < props.dim; ++d) {
        float sum = 0.0f;
        for (int ky = 0; ky < props.kernel_size; ++ky) {
            for (int kx = 0; kx < props.kernel_size; ++kx) {
                const int sample_y = get_window_start(y, props.height, props.kernel_size, neighborhood_size, props.dilation) + ky * props.dilation;
                const int sample_x = get_window_start(x, props.width, props.kernel_size, neighborhood_size, props.dilation) + kx * props.dilation;

                if (sample_y >= 0 && sample_y < props.height && sample_x >= 0 && sample_x < props.width) {
                    const int value_offset = b * props.value_stride_b + h * props.value_stride_h + sample_y * props.value_stride_y + sample_x * props.value_stride_x;
                    const int attn_index = ky * props.kernel_size + kx;
                    sum += attn[attn_offset + attn_index * props.attn_stride_k] * value[value_offset + d * props.value_stride_d];
                }
            }
        }
        output[output_offset + d * props.output_stride_d] = sum;
    }
}
