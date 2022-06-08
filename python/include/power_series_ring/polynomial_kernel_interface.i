%module polynomial_kernel
%{
   #include <power_series_ring/polynomial_kernel_interface.h>
   #include <modulo/modint.h>
   #include <utility>
   #include <exception>
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
%}

%include "stdint.i"
%typemap(out) std::vector<long long> (PyObject* tmp) %{
   tmp = PyList_New($1.size());
   for(int i = 0; i < $1.size(); ++i)
      PyList_SET_ITEM(tmp,i,PyLong_FromLong($1[i]));
   $result = SWIG_Python_AppendOutput($result,tmp);
%}
%typemap(out) std::vector<math::mi> (PyObject* tmp) %{
   tmp = PyList_New($1.size());
   for(int i = 0; i < $1.size(); ++i)
      PyList_SET_ITEM(tmp,i,PyLong_FromUnsignedLong(li1.rv($1[i].get_val())));
   $result = SWIG_Python_AppendOutput($result,tmp);
%}
%typemap(in) ntt_kernel *self(math::lmi li,std::uint32_t mod,int res,void* argp) {
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
      li=$1->get_li();
      mod=$1->get_P();
   }
}
%typemap(in) std::vector<math::mi> ntt_kernel_poly_input1,std::vector<math::mi> ntt_kernel_poly_input2,std::vector<math::mi> ntt_kernel_poly_input3,std::vector<math::mi> ntt_kernel_poly_input4{
   auto &&conv=convert_PyObject_List_to_mi_poly_with_modulo($input,li1,mod1);
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
