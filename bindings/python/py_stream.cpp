#include "py_bindings.h"

namespace {

struct PyStreamGuard {
    std::unique_ptr<StreamGuard> guard;

    explicit PyStreamGuard(const Stream& stream)
        : guard(std::make_unique<StreamGuard>(stream)) {}

    Stream enter() const {
        return current_stream();
    }

    void exit(const py::object&, const py::object&, const py::object&) {
        guard.reset();
    }
};

}  // namespace

void bind_stream(py::module_& m) {
    py::class_<Stream>(m, "Stream")
        .def("synchronize", &Stream::synchronize);

    py::class_<Event>(m, "Event")
        .def(py::init<>())
        .def("synchronize", &Event::synchronize);

    py::class_<PyStreamGuard>(m, "StreamGuard")
        .def("__enter__", &PyStreamGuard::enter)
        .def("__exit__", &PyStreamGuard::exit);

    m.def("current_stream",
          []() {
              return current_stream();
          },
          "Return the thread-local active CUDA stream (always non-default).");

    m.def("set_current_stream",
          [](const Stream& stream) {
              set_current_stream(stream);
          },
          py::arg("stream"),
          "Set the thread-local active CUDA stream.");

    m.def("stream_from_pool",
          [](int idx) {
              return stream_from_pool(idx);
          },
          py::arg("idx"),
          "Get a non-default stream by pool index.");

    m.def("next_stream",
          []() {
              return next_stream();
          },
          "Get the next non-default stream from the runtime pool.");

    m.def("stream_pool_size",
          []() {
              return stream_pool_size();
          },
          "Get runtime stream pool size.");

    m.def("stream_guard",
          [](const Stream& stream) {
              return PyStreamGuard(stream);
          },
          py::arg("stream"),
          "Context manager that temporarily sets current stream.");

    m.def("synchronize",
          []() {
              Stream stream = current_stream();
              stream.synchronize();
          },
          "Synchronize current CUDA stream.");

    m.def("record_event",
          [](Event& event, const Stream& stream) {
              record(event, stream);
          },
          py::arg("event"),
          py::arg("stream"),
          "Record event on stream.");

    m.def("wait_event",
          [](const Stream& stream, Event& event) {
              wait(stream, event);
          },
          py::arg("stream"),
          py::arg("event"),
          "Make stream wait for event.");
}
