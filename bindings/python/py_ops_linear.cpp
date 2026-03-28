#include "py_bindings.h"

namespace {

std::pair<std::shared_ptr<Tensor>, LinearContextPy> linear_forward_py(
    const std::shared_ptr<Tensor>& x,
    const std::shared_ptr<Tensor>& w,
    const std::shared_ptr<Tensor>& b) {
    if (!x || !w) {
        throw std::invalid_argument("ktorch.linear_forward: x and w must not be None.");
    }

    const Tensor* b_ptr = b ? b.get() : nullptr;
    LinearResults out = linear_forward(*x, *w, b_ptr, &py_default_stream(), py_default_cublas_handle());

    LinearContextPy ctx;
    ctx.ctx = out.ctx;
    ctx.x = x;
    ctx.w = w;
    ctx.ctx.X = ctx.x.get();
    ctx.ctx.W = ctx.w.get();

    return {make_tensor_shared(std::move(out.Y)), std::move(ctx)};
}

LinearGradsPy linear_backward_py(
    const std::shared_ptr<Tensor>& dY,
    const LinearContextPy& ctx,
    bool needs_dX,
    bool needs_dW,
    bool needs_db) {
    if (!dY) {
        throw std::invalid_argument("ktorch.linear_backward: dY must not be None.");
    }

    LinearGrads grads = linear_backward(*dY, ctx.ctx, needs_dX, needs_dW, needs_db, &py_default_stream(), py_default_cublas_handle());

    LinearGradsPy out;
    out.dX = optional_tensor_to_shared(std::move(grads.dX));
    out.dW = optional_tensor_to_shared(std::move(grads.dW));
    out.db = optional_tensor_to_shared(std::move(grads.db));
    out.has_dX = grads.has_dX;
    out.has_dW = grads.has_dW;
    out.has_db = grads.has_db;
    return out;
}

}  // namespace

void bind_linear(py::module_& m) {
    py::class_<LinearContextPy>(m, "LinearContext")
        .def_property_readonly("m", [](const LinearContextPy& self) { return self.ctx.m; })
        .def_property_readonly("n", [](const LinearContextPy& self) { return self.ctx.n; })
        .def_property_readonly("k", [](const LinearContextPy& self) { return self.ctx.k; })
        .def_property_readonly("has_bias", [](const LinearContextPy& self) { return self.ctx.has_bias; });

    py::class_<LinearGradsPy>(m, "LinearGrads")
        .def_readonly("dX", &LinearGradsPy::dX)
        .def_readonly("dW", &LinearGradsPy::dW)
        .def_readonly("db", &LinearGradsPy::db)
        .def_readonly("has_dX", &LinearGradsPy::has_dX)
        .def_readonly("has_dW", &LinearGradsPy::has_dW)
        .def_readonly("has_db", &LinearGradsPy::has_db);

    m.def("linear_forward", &linear_forward_py,
          py::arg("x"), py::arg("w"), py::arg("b") = nullptr,
          "Returns (y, ctx).");

    m.def("linear_backward", &linear_backward_py,
          py::arg("dY"), py::arg("ctx"),
          py::arg("needs_dX") = true,
          py::arg("needs_dW") = true,
          py::arg("needs_db") = true);
}

