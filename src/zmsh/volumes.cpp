#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <pybind11/pybind11.h>


PYBIND11_MODULE(volumes, m) {
    m.doc() = R"pbdoc(
        Computing signed volumes
    )pbdoc";

    m.def("add", [](int i, int j) { return i - j; }, R"pbdoc()pbdoc");
}
