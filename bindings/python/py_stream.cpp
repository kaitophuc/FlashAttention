#include "py_bindings.h"

void bind_stream(py::module_& m) {
    py::class_<Stream>(m, "Stream")
        .def("synchronize", &Stream::synchronize);

    m.def("default_stream",
          []() -> Stream& {
              return py_default_stream();
          },
          py::return_value_policy::reference,
          "Return the process-wide default CUDA stream wrapper.");

    m.def("synchronize",
          []() {
              py_default_stream().synchronize();
          },
          "Synchronize the default CUDA stream.");
}
