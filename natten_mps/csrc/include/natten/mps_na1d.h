#pragma once
#include <torch/extension.h>

namespace natten {
namespace mps {

torch::Tensor na1d_forward(
    const torch::Tensor &query,
    const torch::Tensor &key,
    const torch::Tensor &value,
    const torch::Tensor &rpb,
    const int64_t kernel_size,
    const int64_t dilation,
    const int64_t is_causal,
    const int64_t original_length);

} // namespace mps
} // namespace natten
