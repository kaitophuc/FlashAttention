#include "py_bindings.h"

void bind_tensor(py::module_& m) {
    py::class_<Tensor, std::shared_ptr<Tensor>>(m, "Tensor")
        .def(py::init([](const std::vector<int64_t>& shape, DType dtype, Device device) {
            return make_tensor(shape, dtype, device);
        }),
        py::arg("shape"), py::arg("dtype") = DType::F32, py::arg("device") = Device::CUDA)
        .def_property_readonly("shape", [](const Tensor& self) { return self.shape_; })
        .def_property_readonly("strides", [](const Tensor& self) { return self.strides_; })
        .def_property_readonly("dtype", [](const Tensor& self) { return self.dtype_; })
        .def_property_readonly("device", [](const Tensor& self) { return self.device_; })
        .def("numel", &Tensor::numel)
        .def("nbytes", &Tensor::nbytes)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("view", &Tensor::view, py::arg("new_shape"))
        .def("clone", [](const Tensor& self, Device device) { return tensor_clone(self, device); }, py::arg("device"))
        .def("zero_", [](Tensor& self) {
            if (self.device_ == Device::CUDA) {
                self.zero_(py_default_stream());
                py_default_stream().synchronize();
            } else {
                self.zero_();
            }
        })
        .def("copy_from_list_float", &tensor_copy_from_list_float, py::arg("values"))
        .def("copy_from_list_int32", &tensor_copy_from_list_int32, py::arg("values"))
        .def("to_list_float", &tensor_to_list_float)
        .def_static("empty", &tensor_empty, py::arg("shape"), py::arg("dtype") = DType::F32, py::arg("device") = Device::CUDA)
        .def_static("zeros", &tensor_zeros, py::arg("shape"), py::arg("dtype") = DType::F32, py::arg("device") = Device::CUDA);

    m.def("empty", &tensor_empty, py::arg("shape"), py::arg("dtype") = DType::F32, py::arg("device") = Device::CUDA);
    m.def("zeros", &tensor_zeros, py::arg("shape"), py::arg("dtype") = DType::F32, py::arg("device") = Device::CUDA);
}
