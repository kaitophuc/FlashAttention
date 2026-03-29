#include "py_bindings.h"

namespace {

std::shared_ptr<Tensor> classification_correct_count_py(const std::shared_ptr<Tensor>& logits,
                                                        const std::shared_ptr<Tensor>& labels) {
    if (logits == nullptr || labels == nullptr) {
        throw std::invalid_argument("ktorch.classification_correct_count: logits and labels must not be None.");
    }

    Tensor out = classification_correct_count(*logits, *labels, &py_default_stream());
    return make_tensor_shared(std::move(out));
}

}  // namespace

void bind_classification(py::module_& m) {
    m.def("classification_correct_count",
          &classification_correct_count_py,
          py::arg("logits"),
          py::arg("labels"));
}
