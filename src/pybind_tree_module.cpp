#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include "numpy_array_wrapper.hpp"
#include "tree_wrapper.hpp"



namespace py = pybind11;


int example_function(np_array<double> arr)
{
    auto smth = NumpyArrayWrapper<double>(arr);
    auto smth2 = smth.get_1D();

    smth2.coeffRef(1) = 10000;

    return smth2.size();
}



PYBIND11_MODULE(pybind_tree, m)
{
    m.doc() = "pybind11 module of tree with specififc functional";

    m.def("example_function", &example_function, "Bind example function");

    py::class_<TreeRegressorMultiMSEWrapper>(m, "TreeRegressor")
        .def(
            py::init<int, int, double>(),
            py::arg("max_depth") = -1,
            py::arg("min_samples_split") = 2,
            py::arg("min_impurity_decrease") = 0.0
        ).
        def("fit", &TreeRegressorMultiMSEWrapper::fit).
        def("predict", &TreeRegressorMultiMSEWrapper::predict);
}
