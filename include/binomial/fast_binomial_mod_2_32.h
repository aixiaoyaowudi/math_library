/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_BINOMIAL_FAST_BINOMIAL_MOD_2_32_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_BINOMIAL_FAST_BINOMIAL_MOD_2_32_H_

#include <type/basic_typedef.h>

namespace math
{
	namespace binomial
	{
		namespace fast_binomial_mod_2_32
		{
			#include <binomial/constants/coefs_for_fast_binomial_2_32>
			constexpr ull coefs[16][16]=coefs_for_fast_binomial_2_32;
			constexpr ui bases[16]=coefs_for_fast_binomial_2_32_bases;
			ui calc_bju(ui j,ui u);
			ui odd_factorial(ui k);
			ui factorial_odd(ui k);
			ui factorial(ui k);
			ui binomial(ui upper,ui lower);
		}
	}
}

#endif