/*
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <swig/swig_functions.h>

std::pair<std::vector<math::mi>,int> convert_PyObject_List_to_mi_poly_with_modulo(PyObject *input,math::lmi &li,std::uint32_t &mod){
	std::pair<std::vector<math::mi>,int> res={{},0};
	if(!mod){
		PyErr_SetString(PyExc_RuntimeError,"kernel not inited.");
		res.second=1;
		return res;
	}
	if(!PyList_Check(input)){
		PyErr_SetString(PyExc_TypeError,"argument must be list.");
		res.second=1;
		return res;
	}else{
		PyObject *modulo=PyLong_FromUnsignedLong(mod),*cur;
		auto func=modulo->ob_type->tp_as_number->nb_remainder;
		Py_ssize_t len=PyList_Size(input);
		res.first.resize(len);
		for(Py_ssize_t i=0;i<len;++i){
			cur=PyList_GET_ITEM(input,i);
			if(!PyLong_Check(cur)){
				PyErr_SetString(PyExc_TypeError,"Elements must be integers.");
				Py_DECREF(modulo);
				res.second=1;
				return res;
			}else{
				PyObject* cur_num=func(cur,modulo);
				res.first[i]=math::ui2mi(li.v(PyLong_AsUnsignedLong(cur_num)));
				Py_DECREF(cur_num);
			}
		}
		Py_DECREF(modulo);
	}
	return res;
}