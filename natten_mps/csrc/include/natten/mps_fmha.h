#pragma once
#include <torch/extension.h>

namespace natten {
namespace mps {

at::Tensor fmha_forward(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value);

} // namespace mps
} // namespace natten
