/*
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_PYTHON_SWIG_SWIG_FUNCTIONS_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_PYTHON_SWIG_SWIG_FUNCTIONS_H_

#include <vector>
#include <utility>
#include <cstdint>
#include <modulo/modint.h>
#include <Python.h>

std::pair<std::vector<math::mi>,int> convert_PyObject_List_to_mi_poly_with_modulo(PyObject *input,math::lmi &li,std::uint32_t &mod);

#endif