/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_TYPE_BASIC_TYPEDEF_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_TYPE_BASIC_TYPEDEF_H_

#if defined(__INTEL_COMPILER)
#define assume_aligned(a,b) __assume_aligned((a),(b))
#define restrict __restrict
#else
#define assume_aligned(a,b) ((a)=__builtin_assume_aligned((a),(b)))
#define restrict __restrict
#endif

#include <cstdint>

namespace math
{
	typedef std::uint32_t ui;
	typedef std::int32_t i32;
	typedef std::int64_t ll;
	typedef std::uint64_t ull;
	typedef __uint128_t u128;
	typedef __int128 i128;
}

#endif