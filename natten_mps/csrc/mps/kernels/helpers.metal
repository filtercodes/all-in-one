#include <metal_stdlib>
using namespace metal;

inline int get_window_start(
    const int index,
    const int length,
    const int kernel_size,
    const int neighborhood_size,
    const int dilation)
{
    if (dilation <= 1) {
        return max(index - neighborhood_size, 0) + (index + neighborhood_size >= length) * (length - index - neighborhood_size - 1);
    }
    int ni = index - neighborhood_size * dilation;
    if (ni < 0) {
        return index % dilation;
    }
    if (index + neighborhood_size * dilation >= length) {
        const int imodd = index % dilation;
        const int a = (int)(length / dilation) * dilation;
        const int b = length - a;
        if (imodd < b) {
            return length - b + imodd - 2 * neighborhood_size * dilation;
        }
        return a + imodd - kernel_size * dilation;
    }
    return ni;
}

inline int get_pb_start(
    const int index,
    const int length,
    const int kernel_size,
    const int neighborhood_size,
    const int dilation)
{
    if (dilation <= 1) {
        return neighborhood_size + (index < neighborhood_size) * (neighborhood_size - index) + (index + neighborhood_size >= length) * (length - index - 1 - neighborhood_size);
    }
    if (index - neighborhood_size * dilation < 0) {
        return kernel_size - 1 - (int)(index / dilation);
    }
    if (index + neighborhood_size * dilation >= length) {
        return (int)((length - index - 1) / dilation);
    }
    return neighborhood_size;
}

inline void softmax(
    thread float* attn_row,
    const int kernel_size)
{
    float max_val = -FLT_MAX;
    for (int j = 0; j < kernel_size; ++j) {
        max_val = max(max_val, attn_row[j]);
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < kernel_size; ++j) {
        attn_row[j] = exp(attn_row[j] - max_val);
        sum_exp += attn_row[j];
    }

    for (int j = 0; j < kernel_size; ++j) {
        attn_row[j] /= sum_exp;
    }
}


