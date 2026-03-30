#include "py_bindings.h"

namespace {

std::shared_ptr<Tensor> softmax_forward_py(const std::shared_ptr<Tensor>& x) {
    if (x == nullptr) {
        throw std::invalid_argument("ktorch.softmax_forward: x must not be None.");
    }

    Stream stream = py_current_stream();
    Tensor y = softmax_forward(*x, &stream);
    return make_tensor_shared(std::move(y));
}

std::shared_ptr<Tensor> softmax_backward_py(const std::shared_ptr<Tensor>& dY, const std::shared_ptr<Tensor>& y) {
    if (dY == nullptr || y == nullptr) {
        throw std::invalid_argument("ktorch.softmax_backward: dY and y must not be None.");
    }

    Stream stream = py_current_stream();
    SoftmaxGrads grads = softmax_backward(*dY, *y, &stream);
    return make_tensor_shared(std::move(grads.dX));
}

std::pair<std::shared_ptr<Tensor>, SoftmaxCrossEntropyContextPy> softmax_cross_entropy_forward_py(
    const std::shared_ptr<Tensor>& logits,
    const std::shared_ptr<Tensor>& labels) {
    if (logits == nullptr || labels == nullptr) {
        throw std::invalid_argument("ktorch.softmax_cross_entropy_forward: logits and labels must not be None.");
    }

    Stream stream = py_current_stream();
    SoftmaxCrossEntropyResults out = softmax_cross_entropy_forward(*logits, *labels, &stream);
    SoftmaxCrossEntropyContextPy ctx(std::move(out.ctx), labels);
    return {make_tensor_shared(std::move(out.loss)), std::move(ctx)};
}

std::shared_ptr<Tensor> softmax_cross_entropy_backward_py(const SoftmaxCrossEntropyContextPy& ctx) {
    Stream stream = py_current_stream();
    SoftmaxCrossEntropyGrads grads = softmax_cross_entropy_backward(ctx.ctx, &stream);
    return make_tensor_shared(std::move(grads.dX));
}

}  // namespace

void bind_softmax(py::module_& m) {
    m.def("softmax_forward", &softmax_forward_py, py::arg("x"));
    m.def("softmax_backward", &softmax_backward_py, py::arg("dY"), py::arg("y"));
}

void bind_softmax_cross_entropy(py::module_& m) {
    py::class_<SoftmaxCrossEntropyContextPy>(m, "SoftmaxCrossEntropyContext")
        .def_property_readonly("m", [](const SoftmaxCrossEntropyContextPy& self) { return self.ctx.m; })
        .def_property_readonly("n", [](const SoftmaxCrossEntropyContextPy& self) { return self.ctx.n; });

    m.def("softmax_cross_entropy_forward",
          &softmax_cross_entropy_forward_py,
          py::arg("logits"),
          py::arg("labels"),
          "Returns (loss, ctx) where loss is a scalar tensor [1].");
    m.def("softmax_cross_entropy_backward",
          &softmax_cross_entropy_backward_py,
          py::arg("ctx"));
}
