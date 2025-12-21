// myann_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "ann.cpp"

namespace py = pybind11;

PYBIND11_MODULE(myann, m) {
    py::class_<ANN>(m, "ANN")
        .def(py::init<double>(), py::arg("lr"))
        .def("add_linear", &ANN::add_linear, py::arg("in_dim"), py::arg("out_dim"))
        .def("add_sigmoid", &ANN::add_sigmoid)
        .def("update", &ANN::update)
        .def("forward", [](ANN &self, const Eigen::VectorXd &x){
            return self.forward(x);
        })
        .def("backward", &ANN::backward, py::arg("y_true"), py::arg("y_pred"))
        .def("bce", &ANN::BCE);
}
/*
Compile command:
c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python3 -m pybind11 --includes) \
    ./myann_module.cpp \
    -o myann$(python3-config --extension-suffix)
*/