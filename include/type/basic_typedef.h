/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_TYPE_BASIC_TYPEDEF_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_TYPE_BASIC_TYPEDEF_H_

#if defined(__INTEL_COMPILER)
#define assume_aligned(a, b) __assume_aligned ((a), (b))
#define restrict __restrict
#else
#define assume_aligned(a, b) ((a) = __builtin_assume_aligned ((a), (b)))
#define restrict __restrict
#endif

#include <cstdint>

namespace math
{
// typedef std::uint32_t ui;
// typedef std::int32_t i32;
// typedef std::int64_t ll;
// typedef std::uint64_t ull;
// typedef __uint128_t u128;
// typedef __int128 i128;
using i32  = std::int32_t;
using i64  = std::int64_t;
using u32  = std::uint32_t;
using u64  = std::uint64_t;
using ui   = u32;
using ull  = u64;
using u128 = __uint128_t;
using i128 = __int128_t;
using ll   = i64;
}

#endif