#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API at::Tensor fft_irfft2_symint(const at::Tensor & self, at::OptionalSymIntArrayRef s=c10::nullopt, at::IntArrayRef dim={-2,-1}, c10::optional<c10::string_view> norm=c10::nullopt);
TORCH_API at::Tensor & fft_irfft2_symint_out(const at::Tensor & self, at::OptionalSymIntArrayRef s, at::IntArrayRef dim, c10::optional<c10::string_view> norm, at::Tensor & out);
} // namespace native
} // namespace at
