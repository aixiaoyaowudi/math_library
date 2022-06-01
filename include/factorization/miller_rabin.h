/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_FACTORIZATION_MILLER_RABIN_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_FACTORIZATION_MILLER_RABIN_H_

#include <type/basic_typedef.h>

namespace math
{
	namespace factorization
	{
		bool miller_rabin_u32(ui k);
		bool miller_rabin_u64(ull k);
	}
}

#endif