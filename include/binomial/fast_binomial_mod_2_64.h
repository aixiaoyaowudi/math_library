/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_BINOMIAL_FAST_BINOMIAL_MOD_2_64_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_BINOMIAL_FAST_BINOMIAL_MOD_2_64_H_

#include <type/basic_typedef.h>

namespace math
{
	namespace binomial
	{
		namespace fast_binomial_mod_2_64
		{
			ull factorial_odd(ull k);
			ull factorial(ull k);
			ull binomial(ull upper,ull lower);
		}
	}
}

#endif