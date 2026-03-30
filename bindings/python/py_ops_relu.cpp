#include "py_bindings.h"

namespace {

std::pair<std::shared_ptr<Tensor>, ReluContextPy> relu_forward_py(const std::shared_ptr<Tensor>& x) {
    if (x == nullptr) {
        throw std::invalid_argument("ktorch.relu_forward: x must not be None.");
    }

    Stream stream = py_current_stream();
    ReluResults out = relu_forward(*x, stream);

    ReluContextPy ctx;
    ctx.ctx = out.ctx;
    ctx.x = x;
    ctx.ctx.X = ctx.x.get();

    return {make_tensor_shared(std::move(out.Y)), std::move(ctx)};
}

std::shared_ptr<Tensor> relu_backward_py(const std::shared_ptr<Tensor>& dY, const ReluContextPy& ctx) {
    if (dY == nullptr) {
        throw std::invalid_argument("ktorch.relu_backward: dY must not be None.");
    }

    Stream stream = py_current_stream();
    ReluGrads grads = relu_backward(*dY, ctx.ctx, stream);
    return make_tensor_shared(std::move(grads.dX));
}

}  // namespace

void bind_relu(py::module_& m) {
    py::class_<ReluContextPy>(m, "ReluContext");

    m.def("relu_forward", &relu_forward_py, py::arg("x"), "Returns (y, ctx).");
    m.def("relu_backward", &relu_backward_py, py::arg("dY"), py::arg("ctx"));
}
