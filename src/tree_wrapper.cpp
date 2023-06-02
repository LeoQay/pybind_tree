#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


#include <Eigen/Eigen>

#include "tree_wrapper.hpp"
#include "numpy_array_wrapper.hpp"



TreeRegressorMultiMSEWrapper::TreeRegressorMultiMSEWrapper(int max_depth_, int min_samples_split_, FLOAT min_impurity_decrease_) :
TreeRegressorMultiMSE(max_depth_, min_samples_split_, min_impurity_decrease_)
{

}


void TreeRegressorMultiMSEWrapper::fit(
        np_array<double> np_X,
        np_array<double> np_G,
        np_array<double> np_coefs
)
{
    NumpyArrayWrapper<double> X(np_X), G(np_G), coefs(np_coefs);

    auto X_view = X.get_2D();
    auto G_view = G.get_2D();
    auto coefs_view = coefs.get_1D();

    TreeRegressorMultiMSE::fit(X_view, G_view, coefs_view);
}


np_array<double> TreeRegressorMultiMSEWrapper::predict(np_array<double> np_X)
{
    NumpyArrayWrapper<double> X(np_X);
    auto X_view = X.get_2D();

    np_array<double> np_pred(X_view.rows());
    NumpyArrayWrapper<double> pred(np_pred);
    auto pred_view = pred.get_1D();

    TreeRegressorMultiMSE::predict(X_view, pred_view);

    return np_pred;
}
