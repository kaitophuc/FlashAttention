#include "py_bindings.h"

void bind_dtype_device(py::module_& m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .export_values();

    py::enum_<DType>(m, "DType")
        .value("F16", DType::F16)
        .value("BF16", DType::BF16)
        .value("F32", DType::F32)
        .value("I32", DType::I32)
        .value("U8", DType::U8)
        .export_values();
}

