#ifndef NUMPY_ARRAY_WRAPPER_HPP
#define NUMPY_ARRAY_WRAPPER_HPP

#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>


namespace py = pybind11;


template<typename T>
using np_array = py::array_t<T, py::array::c_style | py::array::forcecast>;



template<typename T>
class __attribute__((visibility("default"))) NumpyArrayWrapper
{
public:
    using Matrix = Eigen::Map<Eigen::Matrix<T, -1, -1, Eigen::RowMajor>>;
    using Vector = Eigen::Map<Eigen::VectorX<T>>;

    NumpyArrayWrapper(np_array<T> arr) : arr_(arr) {}
    

    Vector get_1D()
    {
        auto info = arr_.request();

        if (info.ndim != 1)
        {
            throw std::runtime_error("Wrong number of dimensions: must be 1");
        }

        return Vector((T*) info.ptr, info.shape[0]);
    }

    Matrix get_2D()
    {
        auto info = arr_.request();

        if (info.ndim != 2)
        {
            throw std::runtime_error("Wrong number of dimensions: must be 2");
        }

        return Matrix((T*) info.ptr, info.shape[0], info.shape[1]);
    }

private:
    np_array<T> arr_;
};



#endif // NUMPY_ARRAY_WRAPPER_HPP
