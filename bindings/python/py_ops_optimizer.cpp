#include "py_bindings.h"

namespace {

void sgd_update_py(Tensor& param, const Tensor& grad, float lr) {
    sgd_update_(param, grad, lr, &py_default_stream());
}

}  // namespace

void bind_optimizer(py::module_& m) {
    m.def("sgd_update_", &sgd_update_py, py::arg("param"), py::arg("grad"), py::arg("lr"));
}
