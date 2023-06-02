NAME=pybind_tree
CC=g++
CPP_VERSION=c++11
PYTHON_PATH=/usr/include/python3.10

SRC=./src/pybind_tree_module.cpp ./src/tree_wrapper.cpp ./src/tree_regressor.cpp
HDR=./include/tree_wrapper.hpp ./include/tree_regressor.hpp ./include/numpy_array_wrapper.hpp

PYBIND11_INCLUDES=-I/usr/include/python3.10 -I/home/leoqay/second/lib/python3.10/site-packages/pybind11/include # $(python3 -m pybind11 --includes)
MODULE_NAME_SUFFIX=.cpython-310-x86_64-linux-gnu.so # $(python3-config --extension-suffix)

all:
	g++ -O2 -Wall -shared -std=$(CPP_VERSION) -fPIC $(PYBIND11_INCLUDES) -I./include $(SRC) $(HDR) -o $(NAME)$(MODULE_NAME_SUFFIX)

clean:
	rm -f *.o *.so
