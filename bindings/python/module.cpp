#include "py_bindings.h"

PYBIND11_MODULE(_C, m) {
    m.doc() = "ktorch native CUDA bindings";

    bind_dtype_device(m);
    bind_stream(m);
    bind_tensor(m);
    bind_linear(m);
    bind_layernorm(m);
    bind_relu(m);
    bind_softmax(m);
    bind_softmax_cross_entropy(m);
    bind_classification(m);
    bind_optimizer(m);
}
