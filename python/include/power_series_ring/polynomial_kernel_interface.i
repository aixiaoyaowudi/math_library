%module polynomial_kernel
%{
   #include "power_series_ring/polynomial_kernel_interface.h"
   #include <utility>
   #include <exception>
   std::pair<std::vector<std::uint32_t>,int> convert_PyObject_List_to_uint32_t_poly_with_modulo(PyObject *input,std::uint32_t modulo){
      std::pair<std::vector<std::uint32_t>,int> res={{},0};
      if(!PyList_Check(input)){
         PyErr_SetString(PyExc_TypeError,"argument must be list.");
         res.second=1;
         return res;
      }else{
         PyObject *mod=PyLong_FromUnsignedLong(modulo),*cur;
         auto func=mod->ob_type->tp_as_number->nb_remainder;
         Py_ssize_t len=PyList_Size(input);
         res.first.resize(len);
         for(Py_ssize_t i=0;i<len;++i){
            cur=PyList_GET_ITEM(input,i);
            if(!PyLong_Check(cur)){
               PyErr_SetString(PyExc_TypeError,"Elements must be integers.");
               Py_DECREF(mod);
               res.second=1;
               return res;
            }else{
               PyObject* cur_num=func(cur,mod);
               res.first[i]=PyLong_AsUnsignedLong(cur_num);
               Py_DECREF(cur_num);
            }
         }
         Py_DECREF(mod);
      }
      return res;
   }
%}

%include "stdint.i"
%typemap(out) std::vector<long long> (PyObject* tmp) %{
   tmp = PyList_New($1.size());
   for(int i = 0; i < $1.size(); ++i)
      PyList_SET_ITEM(tmp,i,PyLong_FromLong($1[i]));
   $result = SWIG_Python_AppendOutput($result,tmp);
%}
%typemap(out) std::vector<std::uint32_t> (PyObject* tmp) %{
   tmp = PyList_New($1.size());
   for(int i = 0; i < $1.size(); ++i)
      PyList_SET_ITEM(tmp,i,PyLong_FromUnsignedLong($1[i]));
   $result = SWIG_Python_AppendOutput($result,tmp);
%}
%typemap(in) ntt_kernel *self(std::uint32_t tmp,int res,void* argp) {
   argp=NULL;
   res = SWIG_ConvertPtr($input, &argp,$1_descriptor, 0|0);
   if (!SWIG_IsOK(res)) {
      PyErr_SetString(PyExc_ValueError,"self pointer must not be null.");
   }
   $1 = reinterpret_cast<$1_type>(argp);
   if(!$1){
      PyErr_SetString(PyExc_ValueError,"self pointer must not be null.");
      return NULL;
   }else{
      tmp=$1->get_P();
   }
}
%typemap(in) std::vector<std::uint32_t> ntt_kernel_poly_input1,std::vector<std::uint32_t> ntt_kernel_poly_input2,std::vector<std::uint32_t> ntt_kernel_poly_input3,std::vector<std::uint32_t> ntt_kernel_poly_input4{
   auto &&conv=convert_PyObject_List_to_uint32_t_poly_with_modulo($input,tmp1);
   if(conv.second) return NULL;
   $1=conv.first;
}
%exception {
   try{
      $action
   }
   catch(std::exception &e){
      PyErr_SetString(PyExc_RuntimeError,e.what());
      return NULL;
   }
   catch(...){
      PyErr_SetString(PyExc_RuntimeError,"runtime error occurred!");
      return NULL;
   }
}
%include "power_series_ring/polynomial_kernel_interface.h"
