#include "py_bindings.h"

namespace {

std::pair<std::shared_ptr<Tensor>, LayerNormContextPy> layernorm_forward_py(
    const std::shared_ptr<Tensor>& x,
    const std::shared_ptr<Tensor>& gamma,
    const std::shared_ptr<Tensor>& beta,
    float eps) {
    if (!x || !gamma || !beta) {
        throw std::invalid_argument("ktorch.layernorm_forward: x, gamma, beta must not be None.");
    }

    Stream stream = py_current_stream();
    LayerNormResults out = layernorm_forward(*x, *gamma, *beta, eps, stream);

    LayerNormContextPy ctx(std::move(out.ctx), x, gamma);
    return {make_tensor_shared(std::move(out.Y)), std::move(ctx)};
}

LayerNormGradsPy layernorm_backward_py(
    const std::shared_ptr<Tensor>& dY,
    const LayerNormContextPy& ctx,
    bool needs_dX,
    bool needs_dgamma,
    bool needs_dbeta) {
    if (!dY) {
        throw std::invalid_argument("ktorch.layernorm_backward: dY must not be None.");
    }

    Stream stream = py_current_stream();
    LayerNormGrads grads = layernorm_backward(*dY, ctx.ctx, needs_dX, needs_dgamma, needs_dbeta, stream);

    LayerNormGradsPy out;
    out.dX = optional_tensor_to_shared(std::move(grads.dX));
    out.dgamma = optional_tensor_to_shared(std::move(grads.dgamma));
    out.dbeta = optional_tensor_to_shared(std::move(grads.dbeta));
    out.has_dX = grads.has_dX;
    out.has_dgamma = grads.has_dgamma;
    out.has_dbeta = grads.has_dbeta;
    return out;
}

}  // namespace

void bind_layernorm(py::module_& m) {
    py::class_<LayerNormContextPy>(m, "LayerNormContext")
        .def_property_readonly("m", [](const LayerNormContextPy& self) { return self.ctx.m; })
        .def_property_readonly("n", [](const LayerNormContextPy& self) { return self.ctx.n; })
        .def_property_readonly("eps", [](const LayerNormContextPy& self) { return self.ctx.eps; });

    py::class_<LayerNormGradsPy>(m, "LayerNormGrads")
        .def_readonly("dX", &LayerNormGradsPy::dX)
        .def_readonly("dgamma", &LayerNormGradsPy::dgamma)
        .def_readonly("dbeta", &LayerNormGradsPy::dbeta)
        .def_readonly("has_dX", &LayerNormGradsPy::has_dX)
        .def_readonly("has_dgamma", &LayerNormGradsPy::has_dgamma)
        .def_readonly("has_dbeta", &LayerNormGradsPy::has_dbeta);

    m.def("layernorm_forward", &layernorm_forward_py,
          py::arg("x"), py::arg("gamma"), py::arg("beta"), py::arg("eps") = 1e-5f,
          "Returns (y, ctx).");

    m.def("layernorm_backward", &layernorm_backward_py,
          py::arg("dY"), py::arg("ctx"),
          py::arg("needs_dX") = true,
          py::arg("needs_dgamma") = true,
          py::arg("needs_dbeta") = true);
}
