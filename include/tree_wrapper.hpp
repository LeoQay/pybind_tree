#ifndef TREE_WRAPPER_HPP
#define TREE_WRAPPER_HPP


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "numpy_array_wrapper.hpp"

#include "tree_regressor.hpp"


namespace py = pybind11;


class TreeRegressorMultiMSEWrapper : private TreeRegressorMultiMSE
{
public:
    explicit TreeRegressorMultiMSEWrapper(int max_depth_, int min_samples_split_, FLOAT min_impurity_decrease_);

    void fit(np_array<double> np_X, np_array<double> np_G, np_array<double> np_coefs);

    np_array<double> predict(np_array<double> np_X);

private:

    Params params;
};


#endif // TREE_TREE_WRAPPER_HPP
