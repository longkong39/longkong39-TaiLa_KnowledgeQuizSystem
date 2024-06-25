#pragma once
// @generated by torchgen/gen.py from DispatchKeyFunction.h

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace cuda {

TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _flash_attention_forward(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & cum_seq_q, const c10::optional<at::Tensor> & cum_seq_k, int64_t max_q, int64_t max_k, double dropout_p, bool is_causal, bool return_debug_mask, c10::optional<double> scale=c10::nullopt);
TORCH_API ::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> _flash_attention_forward_symint(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const c10::optional<at::Tensor> & cum_seq_q, const c10::optional<at::Tensor> & cum_seq_k, c10::SymInt max_q, c10::SymInt max_k, double dropout_p, bool is_causal, bool return_debug_mask, c10::optional<double> scale=c10::nullopt);

} // namespace cuda
} // namespace at
