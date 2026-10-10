#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

template <typename T>
__global__ void forward_kernel(const T* gate, const T* up, T* output, int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float g = static_cast<float>(gate[i]);
    const float s = 1.0f / (1.0f + expf(-g));
    output[i] = static_cast<T>(g * s * static_cast<float>(up[i]));
  }
}

template <typename T>
__global__ void backward_kernel(const T* gradient, const T* gate, const T* up,
                                T* dgate, T* dup, int64_t count) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float g = static_cast<float>(gate[i]);
    const float s = 1.0f / (1.0f + expf(-g));
    const float grad = static_cast<float>(gradient[i]);
    dgate[i] = static_cast<T>(grad * static_cast<float>(up[i]) * s * (1.0f + g * (1.0f - s)));
    dup[i] = static_cast<T>(grad * g * s);
  }
}

void validate(torch::Tensor gate, torch::Tensor up) {
  TORCH_CHECK(gate.is_cuda() && up.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(gate.device() == up.device(), "Devices must match");
  TORCH_CHECK(gate.sizes() == up.sizes() && gate.scalar_type() == up.scalar_type(), "Shapes/dtypes must match");
  TORCH_CHECK(gate.is_contiguous() && up.is_contiguous(), "Contiguous tensors required");
  TORCH_CHECK(gate.scalar_type() == at::kFloat || gate.scalar_type() == at::kHalf ||
              gate.scalar_type() == at::kBFloat16, "Supported types: FP32/FP16/BF16");
}

torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up) {
  validate(gate, up);
  c10::cuda::CUDAGuard guard(gate.device());
  auto output = torch::empty_like(gate);
  const auto count = gate.numel();
  if (count == 0) return output;
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      gate.scalar_type(), "swiglu_forward", [&] {
    forward_kernel<scalar_t><<<(count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> swiglu_backward(torch::Tensor gradient, torch::Tensor gate, torch::Tensor up) {
  validate(gate, up);
  validate(gate, gradient);
  c10::cuda::CUDAGuard guard(gate.device());
  auto dgate = torch::empty_like(gate), dup = torch::empty_like(up);
  const auto count = gate.numel();
  if (count == 0) return {dgate, dup};
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      gate.scalar_type(), "swiglu_backward", [&] {
    backward_kernel<scalar_t><<<(count + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        gradient.data_ptr<scalar_t>(), gate.data_ptr<scalar_t>(), up.data_ptr<scalar_t>(),
        dgate.data_ptr<scalar_t>(), dup.data_ptr<scalar_t>(), count);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dgate, dup};
}
