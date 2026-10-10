#include <torch/extension.h>

torch::Tensor swiglu_forward(torch::Tensor gate, torch::Tensor up);
std::vector<torch::Tensor> swiglu_backward(torch::Tensor gradient, torch::Tensor gate, torch::Tensor up);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &swiglu_forward);
  module.def("backward", &swiglu_backward);
}
