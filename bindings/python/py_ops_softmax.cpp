#include "py_bindings.h"

namespace {

std::shared_ptr<Tensor> softmax_forward_py(const std::shared_ptr<Tensor>& x) {
    if (x == nullptr) {
        throw std::invalid_argument("ktorch.softmax_forward: x must not be None.");
    }

    Tensor y = softmax_forward(*x, &py_default_stream());
    return make_tensor_shared(std::move(y));
}

std::shared_ptr<Tensor> softmax_backward_py(const std::shared_ptr<Tensor>& dY, const std::shared_ptr<Tensor>& y) {
    if (dY == nullptr || y == nullptr) {
        throw std::invalid_argument("ktorch.softmax_backward: dY and y must not be None.");
    }

    SoftmaxGrads grads = softmax_backward(*dY, *y, &py_default_stream());
    return make_tensor_shared(std::move(grads.dX));
}

}  // namespace

void bind_softmax(py::module_& m) {
    m.def("softmax_forward", &softmax_forward_py, py::arg("x"));
    m.def("softmax_backward", &softmax_backward_py, py::arg("dY"), py::arg("y"));
}

