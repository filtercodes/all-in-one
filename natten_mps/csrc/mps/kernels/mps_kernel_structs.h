#ifndef NATTEN_MPS_KERNEL_STRUCTS_H
#define NATTEN_MPS_KERNEL_STRUCTS_H

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define KERNEL_STRUCT struct
#else
#include <cstdint>
// By removing alignas(16), we allow both the C++ and Metal compilers to use their
// default, natural alignment for these structs. Since they only contain int32_t,
// this alignment will be consistent and correct.
#define KERNEL_STRUCT struct
#endif

// 1D Attention Value
KERNEL_STRUCT NA1dAVProperties {
    int32_t batch_size, heads, length, dim, kernel_size, dilation, is_causal;
    int32_t attn_stride_b, attn_stride_h, attn_stride_l, attn_stride_k;
    int32_t value_stride_b, value_stride_h, value_stride_l, value_stride_d;
    int32_t output_stride_b, output_stride_h, output_stride_l, output_stride_d;
};

// 1D Query-Key-RPB
KERNEL_STRUCT NA1dQkrpbProperties {
    int32_t batch_size, heads, length, dim, kernel_size, dilation, is_causal;
    int32_t query_stride_b, query_stride_h, query_stride_l, query_stride_d;
    int32_t key_stride_b, key_stride_h, key_stride_l, key_stride_d;
    int32_t rpb_stride_h, rpb_stride_l;
    int32_t attn_stride_b, attn_stride_h, attn_stride_l, attn_stride_k;
};

// 1D Permute and Reshape
KERNEL_STRUCT PermuteAndReshape1dProperties {
    int32_t batch_size, heads, length, dim;
    int32_t context_stride_b, context_stride_h, context_stride_l, context_stride_d;
    int32_t output_stride_b, output_stride_l, output_stride_d;
};

// 2D Attention Value
KERNEL_STRUCT NA2dAVProperties {
    int32_t batch_size, heads, height, width, dim, kernel_size, dilation, is_causal;
    int32_t attn_stride_b, attn_stride_h, attn_stride_y, attn_stride_x, attn_stride_k;
    int32_t value_stride_b, value_stride_h, value_stride_y, value_stride_x, value_stride_d;
    int32_t output_stride_b, output_stride_h, output_stride_y, output_stride_x, output_stride_d;
};

// 2D Query-Key-RPB
KERNEL_STRUCT NA2dQkrpbProperties {
    int32_t batch_size, heads, height, width, dim, kernel_size, dilation, is_causal;
    int32_t query_stride_b, query_stride_h, query_stride_y, query_stride_x, query_stride_d;
    int32_t key_stride_b, key_stride_h, key_stride_y, key_stride_x, key_stride_d;
    int32_t rpb_stride_h, rpb_stride_y, rpb_stride_x;
    int32_t attn_stride_b, attn_stride_h, attn_stride_y, attn_stride_x, attn_stride_k;
};

// 2D Permute and Reshape
KERNEL_STRUCT PermuteAndReshape2dProperties {
    int32_t batch_size, heads, height, width, dim;
    int32_t context_stride_b, context_stride_h, context_stride_y, context_stride_x, context_stride_d;
    int32_t output_stride_b, output_stride_y, output_stride_x, output_stride_d;
};

// 3D Attention Value
KERNEL_STRUCT NA3dAVProperties {
    int32_t batch_size, heads, depth, height, width, dim;
    int32_t kernel_size_d, kernel_size, dilation_d, dilation, is_causal;
    int32_t attn_stride_b, attn_stride_h, attn_stride_z, attn_stride_y, attn_stride_x, attn_stride_k;
    int32_t value_stride_b, value_stride_h, value_stride_z, value_stride_y, value_stride_x, value_stride_d;
    int32_t output_stride_b, output_stride_h, output_stride_z, output_stride_y, output_stride_x, output_stride_d;
};

// 3D Query-Key-RPB
KERNEL_STRUCT NA3dQkrpbProperties {
    int32_t batch_size, heads, depth, height, width, dim;
    int32_t kernel_size_d, kernel_size, dilation_d, dilation, is_causal;
    int32_t query_stride_b, query_stride_h, query_stride_z, query_stride_y, query_stride_x, query_stride_d;
    int32_t key_stride_b, key_stride_h, key_stride_z, key_stride_y, key_stride_x, key_stride_d;
    int32_t rpb_stride_h, rpb_stride_z, rpb_stride_y, rpb_stride_x;
    int32_t attn_stride_b, attn_stride_h, attn_stride_z, attn_stride_y, attn_stride_x, attn_stride_k;
};

// 3D Permute and Reshape
KERNEL_STRUCT PermuteAndReshape3dProperties {
    int32_t batch_size, heads, depth, height, width, dim;
    int32_t context_stride_b, context_stride_h, context_stride_z, context_stride_y, context_stride_x, context_stride_d;
    int32_t output_stride_b, output_stride_z, output_stride_y, output_stride_x, output_stride_d;
};

#endif // NATTEN_MPS_KERNEL_STRUCTS_H