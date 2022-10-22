/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_MODULO_MODINT_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_MODULO_MODINT_H_

#include <cstdio>
#include <immintrin.h>
#include <iostream>
#include <type/basic_typedef.h>
#include <type_traits>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace math
{
namespace modulo
{
class barrett_reduction_u32
{
private:
  u64 b, m, r;

public:
  barrett_reduction_u32 (u64 p)
      : b (p), m ((u64) ((u128 (1) << 64) / p)),
        r ((u64) (((u64 (1) << 63) + p - 1) / p * p))
  {
  }
  barrett_reduction_u32 () {}
  barrett_reduction_u32 (const barrett_reduction_u32 &d)
      : b (d.b), m (d.m), r (d.r)
  {
  }
  void
  init (u64 p)
  {
    b = p;
    m = (u64) ((u128 (1) << 64) / p);
    r = (u64) (((u64 (1) << 63) + p - 1) / p * p);
  }
  template <typename T>
  typename std::enable_if<std::is_integral_v<T> && std::is_unsigned_v<T>,
                          u64>::type
  reduce (const T &a) const
  {
    u64 k = static_cast<u64> (a);
    u64 q = (u64) ((u128 (m) * k) >> 64);
    u64 t = k - q * b;
    return t >= b ? t - b : t;
  }
  template <typename T>
  typename std::enable_if<std::is_integral_v<T> && std::is_signed_v<T>,
                          i64>::type
  reduce (const T &a) const
  {
    if (a >= 0)
      return static_cast<i64> (this->reduce (static_cast<u64> (a)));
    else
      return static_cast<i64> (this->reduce (static_cast<u64> (a) + r));
  }
};
#define INLINE_OP __attribute__ ((__always_inline__))
namespace modint
{
extern barrett_reduction_u32 global_barrett_reduction_u32;
constexpr u32 default_mod = 998244353;
struct montgomery_modint_lib
{
  constexpr static u32
  calc_k (u32 mod, u32 len)
  {
    u32 ans = 1;
    for (u32 i = 1; i < len; ++i)
      ans = (ans * (mod + 1) + 1);
    return ans;
  }
  u32 P, P2, NP, Pk, Pr;
  barrett_reduction_u32 barrett;
  static constexpr u32 u32_len = sizeof (u32) * 8;
  montgomery_modint_lib (u32 P0)
      : P (P0), P2 (P0 * 2), NP ((-u64 (P0)) % P0), Pk (calc_k (P0, u32_len)),
        Pr (((u32 (1) << 31) + P0 - 1) / P0 * P0), barrett (P0)
  {
  }
  montgomery_modint_lib (const montgomery_modint_lib &d)
      : P (d.P), P2 (d.P2), NP (d.NP), Pk (d.Pk), Pr (d.Pr),
        barrett (d.barrett)
  {
  }
  montgomery_modint_lib () {}
  INLINE_OP u32
  redd (u32 k) const
  {
    return k >= P2 ? k - P2 : k;
  }
  INLINE_OP u32
  reds (u32 k) const
  {
    return k >= P ? k - P : k;
  }
  INLINE_OP u32
  redu (u64 k) const
  {
    return (k + u64 (u32 (k) * Pk) * P) >> u32_len;
  }
  INLINE_OP u32
  add (u32 a, u32 b) const
  {
    return redd (a + b);
  }
  INLINE_OP u32
  sub (u32 a, u32 b) const
  {
    return redd (a - b + P2);
  }
  INLINE_OP u32
  mul (u32 a, u32 b) const
  {
    return redu (u64 (a) * b);
  }
  INLINE_OP u32
  neg (u32 a) const
  {
    return redd (P2 - a);
  }
  INLINE_OP u32
  to_modint (u32 a) const
  {
    return redu (u64 (a) * NP);
  }
  INLINE_OP u32
  to_int (u32 a) const
  {
    return reds (redu (a));
  }
  INLINE_OP u32
  v (u32 a) const
  {
    return to_modint (a);
  }
  INLINE_OP u32
  rv (u32 a) const
  {
    return to_int (a);
  }
};
struct montgomery_modint
{
  static montgomery_modint_lib mlib;
#if defined(_OPENMP)
#pragma omp threadprivate(mlib)
#endif
  u32 val;
  void
  init (u32 a)
  {
    val = mlib.redu (u64 (a) * mlib.NP);
  }
  montgomery_modint () { val = 0; }
  montgomery_modint (const montgomery_modint &a) : val (a.val) {}
  template <typename T,
            typename std::enable_if<sizeof (T) <= sizeof (i32)
                                    && std::is_signed_v<T> >::type *_dummy_ptr
            = nullptr>
  montgomery_modint (const T &v)
      : val (mlib.redu (u64 ((v >= 0) ? v : (static_cast<u32> (v) + mlib.Pr))
                        * mlib.NP))
  {
  }
  template <typename T, typename std::enable_if<
                            sizeof (T) <= sizeof (u32)
                            && std::is_unsigned_v<T> >::type *_dummy_ptr
                        = nullptr>
  montgomery_modint (const T &v) : val (mlib.redu (u64 (v) * mlib.NP))
  {
  }
  template <typename T,
            typename std::enable_if<
                sizeof (T) <= sizeof (u64) && (sizeof (T) > sizeof (u32))
                && std::is_integral_v<T> >::type *_dummy_ptr
            = nullptr>
  montgomery_modint (const T &v)
      : val (mlib.redu (u64 (mlib.barrett.reduce (v)) * mlib.NP))
  {
  }
  montgomery_modint &
  operator= (const montgomery_modint &b)
  {
    val = b.val;
    return *this;
  }
  montgomery_modint &
  operator+= (const montgomery_modint &b)
  {
    val = mlib.redd (val + b.val);
    return *this;
  }
  montgomery_modint &
  operator-= (const montgomery_modint &b)
  {
    val = mlib.redd (val - b.val + mlib.P2);
    return *this;
  }
  montgomery_modint
  operator+ (const montgomery_modint &b) const
  {
    return montgomery_modint (*this) += b;
  }
  montgomery_modint
  operator- (const montgomery_modint &b) const
  {
    return montgomery_modint (*this) -= b;
  }
  montgomery_modint &
  operator*= (const montgomery_modint &b)
  {
    val = mlib.redu (u64 (val) * b.val);
    return *this;
  }
  montgomery_modint
  operator* (const montgomery_modint &b) const
  {
    return montgomery_modint (*this) *= b;
  }
  montgomery_modint
  operator- () const
  {
    montgomery_modint b;
    return b -= (*this);
  }
  u32
  real_val () const
  {
    return mlib.reds (mlib.redu (val));
  }
  u32
  get_val () const
  {
    return val;
  }
  friend std::istream &
  operator>> (std::istream &in, montgomery_modint &m_int)
  {
    u32 inp;
    in >> inp;
    m_int.init (inp);
    return in;
  }
  friend std::ostream &
  operator<< (std::ostream &out, const montgomery_modint &m_int)
  {
    out << m_int.real_val ();
    return out;
  }
};
static INLINE_OP inline montgomery_modint
ui2mi (u32 k)
{
  montgomery_modint mval;
  mval.val = k;
  return mval;
}
struct montgomery_modlongint_lib
{
  constexpr static u64
  calc_k (u64 mod, u64 len)
  {
    u64 ans = 1;
    for (u32 i = 1; i < len; ++i)
      ans = (ans * (mod + 1) + 1);
    return ans;
  }
  u64 P, P2, NP, Pk, Pr;
  static constexpr u32 u64_len = sizeof (u64) * 8;
  montgomery_modlongint_lib (u64 P0)
      : P (P0), P2 (P0 * 2), NP ((-u128 (P0)) % P0), Pk (calc_k (P0, u64_len)),
        Pr (((u64 (1) << 63) + P0 - 1) % P0)
  {
  }
  montgomery_modlongint_lib (const montgomery_modlongint_lib &d)
      : P (d.P), P2 (d.P2), NP (d.NP), Pk (d.Pk), Pr (d.Pr)
  {
  }
  montgomery_modlongint_lib () {}
  INLINE_OP u64
  redd (u64 k) const
  {
    return k >= P2 ? k - P2 : k;
  }
  INLINE_OP u64
  reds (u64 k) const
  {
    return k >= P ? k - P : k;
  }
  INLINE_OP u64
  redu (u128 k) const
  {
    return (k + u128 (u64 (k) * Pk) * P) >> u64_len;
  }
  INLINE_OP u64
  add (u64 a, u64 b) const
  {
    return redd (a + b);
  }
  INLINE_OP u64
  sub (u64 a, u64 b) const
  {
    return redd (a - b + P2);
  }
  INLINE_OP u64
  mul (u64 a, u64 b) const
  {
    return redu (u128 (a) * b);
  }
  INLINE_OP u64
  neg (u64 a) const
  {
    return redd (P2 - a);
  }
  INLINE_OP u64
  to_modint (u64 a) const
  {
    return redu (u128 (a) * NP);
  }
  INLINE_OP u64
  to_int (u64 a) const
  {
    return reds (redu (a));
  }
  INLINE_OP u64
  v (u64 a) const
  {
    return to_modint (a);
  }
  INLINE_OP u64
  rv (u64 a) const
  {
    return to_int (a);
  }
};
struct montgomery_modlongint
{
  static montgomery_modlongint_lib mlib;
#if defined(_OPENMP)
#pragma omp threadprivate(mlib)
#endif
  u64 val;
  void
  init (u64 a)
  {
    val = mlib.redu (u128 (a) * mlib.NP);
  }
  montgomery_modlongint () { val = 0; }
  montgomery_modlongint (const montgomery_modlongint &a) : val (a.val) {}
  template <typename T,
            typename std::enable_if<std::is_signed_v<T> >::type *_dummy_ptr
            = nullptr>
  montgomery_modlongint (const T &v)
      : val (mlib.redu (u128 ((v >= 0) ? v : (static_cast<u64> (v) + mlib.Pr))
                        * mlib.NP))
  {
  }
  template <typename T,
            typename std::enable_if<std::is_unsigned_v<T> >::type *_dummy_ptr
            = nullptr>
  montgomery_modlongint (const T &v)
      : val (mlib.redu (static_cast<u128> (v) * mlib.NP))
  {
  }
  montgomery_modlongint &
  operator= (const montgomery_modlongint &b)
  {
    val = b.val;
    return *this;
  }
  montgomery_modlongint &
  operator+= (const montgomery_modlongint &b)
  {
    val = mlib.redd (val + b.val);
    return *this;
  }
  montgomery_modlongint &
  operator-= (const montgomery_modlongint &b)
  {
    val = mlib.redd (val - b.val + mlib.P2);
    return *this;
  }
  montgomery_modlongint
  operator+ (const montgomery_modlongint &b) const
  {
    return montgomery_modlongint (*this) += b;
  }
  montgomery_modlongint
  operator- (const montgomery_modlongint &b) const
  {
    return montgomery_modlongint (*this) -= b;
  }
  montgomery_modlongint &
  operator*= (const montgomery_modlongint &b)
  {
    val = mlib.redu (u128 (val) * b.val);
    return *this;
  }
  montgomery_modlongint
  operator* (const montgomery_modlongint &b) const
  {
    return montgomery_modlongint (*this) *= b;
  }
  montgomery_modlongint
  operator- () const
  {
    montgomery_modlongint b;
    return b -= (*this);
  }
  u64
  real_val () const
  {
    return mlib.reds (mlib.redu (val));
  }
  u64
  get_val () const
  {
    return val;
  }
  friend std::istream &
  operator>> (std::istream &in, montgomery_modlongint &m_int)
  {
    u64 inp;
    in >> inp;
    m_int.init (inp);
    return in;
  }
  friend std::ostream &
  operator<< (std::ostream &out, const montgomery_modlongint &m_int)
  {
    out << m_int.real_val ();
    return out;
  }
};
static INLINE_OP inline montgomery_modlongint
ull2mli (u64 k)
{
  montgomery_modlongint mval;
  mval.val = k;
  return mval;
}
using mi  = montgomery_modint;
using mli = montgomery_modlongint;
using lmi = montgomery_modint_lib;
using lml = montgomery_modlongint_lib;
#if defined(__AVX__) && defined(__AVX2__)
struct montgomery_mm256_lib
{
  alignas (32) __m256i P, P2, NP, Pk;
  static constexpr u32 u32_len = sizeof (u32) * 8;
  montgomery_mm256_lib (u32 P0)
  {
    P  = _mm256_set1_epi32 (P0);
    P2 = _mm256_set1_epi32 (P0 * 2);
    NP = _mm256_set1_epi32 (u32 ((-u64 (P0)) % P0));
    Pk = _mm256_set1_epi32 (montgomery_modint_lib::calc_k (P0, u32_len));
  }
  montgomery_mm256_lib () {}
  INLINE_OP __m256i
  redd (__m256i k)
  {
    __m256i a = _mm256_sub_epi32 (k, P2);
    __m256i b = _mm256_and_si256 (
        _mm256_cmpgt_epi32 (_mm256_setzero_si256 (), a), P2);
    return _mm256_add_epi32 (a, b);
  }
  INLINE_OP __m256i
  reds (__m256i k)
  {
    __m256i a = _mm256_sub_epi32 (k, P);
    __m256i b = _mm256_and_si256 (
        _mm256_cmpgt_epi32 (_mm256_setzero_si256 (), a), P);
    return _mm256_add_epi32 (a, b);
  }
  INLINE_OP __m256i
  redu (__m256i k)
  {
    return _mm256_srli_epi64 (
        _mm256_add_epi64 (_mm256_mul_epu32 (_mm256_mul_epu32 (k, Pk), P), k),
        32);
  }
  INLINE_OP __m256i
  mul (__m256i k1, __m256i k2)
  {
    return _mm256_or_si256 (
        redu (_mm256_mul_epu32 (k1, k2)),
        _mm256_slli_epi64 (
            redu (_mm256_mul_epu32 (_mm256_srli_epi64 (k1, 32),
                                    _mm256_srli_epi64 (k2, 32))),
            32));
  }
  INLINE_OP __m256i
  add (__m256i k1, __m256i k2)
  {
    return redd (_mm256_add_epi32 (k1, k2));
  }
  INLINE_OP __m256i
  sub (__m256i k1, __m256i k2)
  {
    return redd (_mm256_add_epi32 (P2, _mm256_sub_epi32 (k1, k2)));
  }
};
using lma = montgomery_mm256_lib;
#endif
#if defined(__AVX512F__) && defined(__AVX512DQ__)
struct montgomery_mm512_lib
{
  alignas (64) __m512i P, P2, NP, Pk;
  static constexpr u32 u32_len = sizeof (u32) * 8;
  montgomery_mm512_lib (u32 P0)
  {
    P  = _mm512_set1_epi32 (P0);
    P2 = _mm512_set1_epi32 (P0 * 2);
    NP = _mm512_set1_epi32 (u32 ((-u64 (P0)) % P0));
    Pk = _mm512_set1_epi32 (montgomery_modint_lib::calc_k (P0, u32_len));
  }
  montgomery_mm512_lib () {}
  INLINE_OP __m512i
  redd (__m512i k)
  {
    __m512i a = _mm512_sub_epi32 (k, P2);
    return _mm512_mask_add_epi32 (a, _mm512_movepi32_mask (a), a, P2);
  }
  INLINE_OP __m512i
  reds (__m512i k)
  {
    __m512i a = _mm512_sub_epi32 (k, P);
    return _mm512_mask_add_epi32 (a, _mm512_movepi32_mask (a), a, P);
  }
  INLINE_OP __m512i
  redu (__m512i k)
  {
    return _mm512_srli_epi64 (
        _mm512_add_epi64 (_mm512_mul_epu32 (_mm512_mul_epu32 (k, Pk), P), k),
        32);
  }
  INLINE_OP __m512i
  mul (__m512i k1, __m512i k2)
  {
    return _mm512_or_si512 (
        redu (_mm512_mul_epu32 (k1, k2)),
        _mm512_slli_epi64 (
            redu (_mm512_mul_epu32 (_mm512_srli_epi64 (k1, 32),
                                    _mm512_srli_epi64 (k2, 32))),
            32));
  }
  INLINE_OP __m512i
  add (__m512i k1, __m512i k2)
  {
    return redd (_mm512_add_epi32 (k1, k2));
  }
  INLINE_OP __m512i
  sub (__m512i k1, __m512i k2)
  {
    return redd (_mm512_add_epi32 (P2, _mm512_sub_epi32 (k1, k2)));
  }
};
using lm5 = montgomery_mm512_lib;
#endif
extern u32 global_mod_mi;
extern u64 global_mod_mli;
#if defined(_OPENMP)
#pragma omp threadprivate(global_mod_mi)
#pragma omp threadprivate(global_mod_mli)
#endif
void set_mod_mi (u32 p);
void set_mod_mli (u64 p);
void set_mod_for_all_threads_mi (u32 p);
void set_mod_for_all_threads_mli (u64 p);
}
#undef INLINE_OP
}
using modulo::modint::default_mod;
using modulo::modint::global_mod_mi;
using modulo::modint::global_mod_mli;
using modulo::modint::lmi;
using modulo::modint::lml;
using modulo::modint::mi;
using modulo::modint::mli;
using modulo::modint::set_mod_mi;
using modulo::modint::set_mod_mli;
using modulo::modint::ui2mi;
using modulo::modint::ull2mli;
#if defined(__AVX__) && defined(__AVX2__)
using modulo::modint::lma;
#endif
#if defined(__AVX512F__) && defined(__AVX512DQ__)
using modulo::modint::lm5;
#endif
using modulo::barrett_reduction_u32;
using fast_mod_32 = barrett_reduction_u32;
}

#endif