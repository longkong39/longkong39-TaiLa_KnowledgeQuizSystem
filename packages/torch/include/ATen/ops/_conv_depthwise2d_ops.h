#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API _conv_depthwise2d_out {
  using schema = const at::Tensor & (const at::Tensor &, const at::Tensor &, c10::SymIntArrayRef, const c10::optional<at::Tensor> &, c10::SymIntArrayRef, c10::SymIntArrayRef, c10::SymIntArrayRef, const at::Tensor &);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_conv_depthwise2d")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "out")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_conv_depthwise2d.out(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias, SymInt[2] stride, SymInt[2] padding, SymInt[2] dilation, *, Tensor(a!) out) -> Tensor(a!)")
  static const at::Tensor & call(const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, const at::Tensor & out);
  static const at::Tensor & redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, const at::Tensor & out);
};

struct TORCH_API _conv_depthwise2d {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, c10::SymIntArrayRef, const c10::optional<at::Tensor> &, c10::SymIntArrayRef, c10::SymIntArrayRef, c10::SymIntArrayRef);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_conv_depthwise2d")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_conv_depthwise2d(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias, SymInt[2] stride, SymInt[2] padding, SymInt[2] dilation) -> Tensor")
  static at::Tensor call(const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, const at::Tensor & weight, c10::SymIntArrayRef kernel_size, const c10::optional<at::Tensor> & bias, c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation);
};

}} // namespace at::_ops