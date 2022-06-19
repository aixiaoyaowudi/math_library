%module polynomial_kernel
%{
    #include <power_series_ring/polynomial_kernel_interface.h>
    #include <modulo/modint.h>
    #include <utility>
    #include <exception>
    #include <swig/swig_functions.h>
%}

%include "stdint.i"

%typemap(out) std::vector<math::mi> (PyObject* tmp)
{
    tmp = PyList_New($1.size());
    for(int i = 0; i < $1.size(); ++i)
        PyList_SET_ITEM(tmp,i,PyLong_FromUnsignedLong(li1.rv($1[i].get_val())));
    $result = SWIG_Python_AppendOutput($result,tmp);
}

%typemap(in) ntt_kernel *self(math::lmi li,std::uint32_t mod,int res,void* argp)
{
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

%apply ntt_kernel *self{ mtt_kernel *self };

%typemap(in) std::vector<math::mi> ntt_kernel_poly_input1,std::vector<math::mi> ntt_kernel_poly_input2,std::vector<math::mi> mtt_kernel_poly_input1,std::vector<math::mi> mtt_kernel_poly_input2
{
    auto &&conv=convert_PyObject_List_to_mi_poly_with_modulo($input,li1,mod1);
    if(conv.second) return NULL;
    $1=conv.first;
}

%exception
{
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

#define SWIG_NOT_EXPORT(x)
%include "power_series_ring/polynomial_kernel_interface.h"
#undef SWIG_NOT_EXPORT

%extend ntt_kernel {
    %pythoncode
    %{
        def __repr__(self):
            if self.get_P()!=0:
                return "Power Series Ring over Ring of integers modulo " + str(self.get_P())
            else:
                return "Power Series Ring over finite field"
    %}
};