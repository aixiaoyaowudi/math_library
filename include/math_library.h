/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_
#define _XIAOYAOWUDI_MATH_LIBRARY_

#if __cplusplus < 201703L
#error "Require C++17 to Compile."
#endif

#if (!defined(__INTEL_COMPILER)) && (!defined(__GNUC__))
#error "Only GNU compiler and Intel compiler is supported."
#endif

#include <type/basic_typedef.h>
#include <tools/tools.h>
#include <type/type.h>
#include <modulo/modulo.h>
#include <binomial/binomial.h>
#include <basic/basic.h>
#include <power_series_ring/power_series_ring.h>
#include <factorization/factorization.h>
#include <init/init.h>

#endif