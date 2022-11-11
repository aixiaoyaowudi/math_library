/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <algorithm>
#include <basic/basic.h>
#include <cassert>
#include <chrono>
#include <cstring>
#include <factorization/factorization.h>
#include <immintrin.h>
#include <power_series_ring/polynomial_kernel.h>
#include <random>
#include <type/basic_typedef.h>

#define NTT_partition_size 10

namespace math
{
#if defined(__AVX__) && defined(__AVX2__)
#define USE_AVX2 1
#endif

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define USE_AVX512 1
#endif

#if USE_AVX2

#define L_INPUT la, li
#define L_PARAM const lma &la, const lmi &li

#else

#define L_INPUT li
#define L_PARAM const lmi &li

#endif

static void
MUL_B_TO_A (u32 _len, u32 *_pt1, u32 *_pt2, L_PARAM)
{
#if USE_AVX2
  if (_len <= 4)
    {
      for (u32 _i = 0; _i < _len; ++_i)
        {
          _pt1[_i] = li.mul (_pt1[_i], _pt2[_i]);
        }
    }
  else
    {
      __m256i *_mapt1 = (__m256i *)(_pt1), *_mapt2 = (__m256i *)(_pt2);
      _len >>= 3;
      for (u32 _i = 0; _i < _len; ++_i, ++_mapt1, ++_mapt2)
        {
          (*_mapt1) = la.mul ((*_mapt1), (*_mapt2));
        }
    }
#else
  for (u32 _i = 0; _i < _len; ++_i)
    {
      _pt1[_i] = li.mul (_pt1[_i], _pt2[_i]);
    }
#endif
}

static void
ADD_B_TO_A (u32 _len, u32 *_pt1, u32 *_pt2, L_PARAM)
{
#if USE_AVX2
  if (_len <= 4)
    {
      for (u32 _i = 0; _i < _len; ++_i)
        {
          _pt1[_i] = li.add (_pt1[_i], _pt2[_i]);
        }
    }
  else
    {
      __m256i *_mapt1 = (__m256i *)(_pt1), *_mapt2 = (__m256i *)(_pt2);
      _len >>= 3;
      for (u32 _i = 0; _i < _len; ++_i, ++_mapt1, ++_mapt2)
        {
          (*_mapt1) = la.add ((*_mapt1), (*_mapt2));
        }
    }
#else
  for (u32 _i = 0; _i < _len; ++_i)
    {
      _pt1[_i] = li.add (_pt1[_i], _pt2[_i]);
    }
#endif
}

static void
SUB_B_FROM_A (u32 _len, u32 *_pt1, u32 *_pt2, L_PARAM)
{
#if USE_AVX2
  if (_len <= 4)
    {
      for (u32 _i = 0; _i < _len; ++_i)
        {
          _pt1[_i] = li.sub (_pt1[_i], _pt2[_i]);
        }
    }
  else
    {
      __m256i *_mapt1 = (__m256i *)(_pt1), *_mapt2 = (__m256i *)(_pt2);
      _len >>= 3;
      for (u32 _i = 0; _i < _len; ++_i, ++_mapt1, ++_mapt2)
        {
          (*_mapt1) = la.sub ((*_mapt1), (*_mapt2));
        }
    }
#else
  for (u32 _i = 0; _i < _len; ++_i)
    {
      _pt1[_i] = li.sub (_pt1[_i], _pt2[_i]);
    }
#endif
}

static void
MUL_B_TIMES_C_TO_A (u32 _len, u32 *_pt1, u32 *_pt2, u32 *_pt3, L_PARAM)
{
#if USE_AVX2
  if (_len <= 4)
    {
      for (u32 _i = 0; _i < _len; ++_i)
        {
          _pt1[_i] = li.mul (_pt2[_i], _pt3[_i]);
        }
    }
  else
    {
      __m256i *_mapt1 = (__m256i *)(_pt1), *_mapt2 = (__m256i *)(_pt2),
              *_mapt3 = (__m256i *)(_pt3);
      _len >>= 3;
      for (u32 _i = 0; _i < _len; ++_i, ++_mapt1, ++_mapt2, ++_mapt3)
        {
          (*_mapt1) = la.mul ((*_mapt2), (*_mapt3));
        }
    }
#else
  for (u32 _i = 0; _i < _len; ++_i)
    {
      _pt1[_i] = li.mul (_pt2[_i], _pt3[_i]);
    }
#endif
}

static void
ADD_B_TIMES_C_TO_A (u32 _len, u32 *_pt1, u32 *_pt2, u32 _val, L_PARAM)
{
#if USE_AVX2
  if (_len <= 4)
    {
      for (u32 _i = 0; _i < _len; ++_i)
        {
          _pt1[_i] = li.add (_pt1[_i], li.mul (_pt2[_i], _val));
        }
    }
  else
    {
      __m256i *_mapt1 = (__m256i *)(_pt1), *_mapt2 = (__m256i *)(_pt2),
              _w = _mm256_set1_epi32 (_val);
      _len >>= 3;
      for (u32 _i = 0; _i < _len; ++_i, ++_mapt1, ++_mapt2)
        {
          (*_mapt1) = la.add ((*_mapt1), la.mul ((*_mapt2), _w));
        }
    }
#else
  for (u32 _i = 0; _i < _len; ++_i)
    {
      _pt1[_i] = li.add (_pt1[_i], li.mul (_pt2[_i], _val));
    }
#endif
}

static void
MUL_B_TIMES_B_TO_A (u32 _len, u32 *_pt1, u32 *_pt2, L_PARAM)
{
#if USE_AVX2
  if (_len <= 4)
    {
      for (u32 _i = 0; _i < _len; ++_i)
        {
          _pt1[_i] = li.mul (_pt1[_i], li.mul (_pt2[_i], _pt2[_i]));
        }
    }
  else
    {
      __m256i *_mapt1 = (__m256i *)(_pt1), *_mapt2 = (__m256i *)(_pt2);
      _len >>= 3;
      for (u32 _i = 0; _i < _len; ++_i, ++_mapt1, ++_mapt2)
        {
          (*_mapt1) = la.mul ((*_mapt1), la.mul ((*_mapt2), (*_mapt2)));
        }
    }
#else
  for (u32 _i = 0; _i < _len; ++_i)
    {
      _pt1[_i] = li.mul (_pt1[_i], li.mul (_pt2[_i], _pt2[_i]));
    }
#endif
}

static void
NEG_B_TO_A (u32 _len, u32 *_pt1, u32 *_pt2, L_PARAM)
{
#if USE_AVX2
  if (_len <= 4)
    {
      for (u32 _i = 0; _i < _len; ++_i)
        {
          _pt1[_i] = li.neg (_pt2[_i]);
        }
    }
  else
    {
      __m256i *_mapt1 = (__m256i *)(_pt1), *_mapt2 = (__m256i *)(_pt2);
      _len >>= 3;
      for (u32 _i = 0; _i < _len; ++_i, ++_mapt1, ++_mapt2)
        {
          (*_mapt1) = la.sub (_mm256_setzero_si256 (), (*_mapt2));
        }
    }
#else
  for (u32 _i = 0; _i < _len; ++_i)
    {
      _pt1[_i] = li.neg (_pt2[_i]);
    }
#endif
}

void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::trick_mul (
    u32 _len, u32 *_pt1, u32 *_pt2, u32 *_pt3, u32 *_pt4, u32 *_pt5, u32 *_pt6)
{
#if USE_AVX2
  if (_len <= 4)
    {
      for (u32 _i = 0; _i < _len; ++_i)
        {
          _pt5[_i] = li.mul (_pt1[_i], _pt3[_i]),
          _pt6[_i] = li.add (li.mul (_pt1[_i], _pt4[_i]),
                             li.mul (_pt2[_i], _pt3[_i]));
        }
    }
  else
    {
      __m256i *_mapt1 = (__m256i *)(_pt1), *_mapt2 = (__m256i *)(_pt2);
      __m256i *_mapt3 = (__m256i *)(_pt3), *_mapt4 = (__m256i *)(_pt4);
      __m256i *_mapt5 = (__m256i *)(_pt5), *_mapt6 = (__m256i *)(_pt6);
      for (u32 _i = 0; _i < (_len >> 3); ++_i)
        {
          _mapt5[_i] = la.mul (_mapt1[_i], _mapt3[_i]),
          _mapt6[_i] = la.add (la.mul (_mapt1[_i], _mapt4[_i]),
                               la.mul (_mapt2[_i], _mapt3[_i]));
        }
    }
#else
  for (u32 _i = 0; _i < _len; ++_i)
    {
      _pt5[_i] = li.mul (_pt1[_i], _pt3[_i]),
      _pt6[_i]
          = li.add (li.mul (_pt1[_i], _pt4[_i]), li.mul (_pt2[_i], _pt3[_i]));
    }
#endif

  dit (_pt5, __builtin_ctz (_len));
  dit (_pt6, __builtin_ctz (_len));

  ADD_B_TO_A ((_len >> 1), _pt5 + (_len >> 1), _pt6, L_INPUT);
}

void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::internal_mul (
    u32 *src1, u32 *src2, u32 *dst, u32 m)
{
  dif (src1, m);
  dif (src2, m);
  MUL_B_TIMES_C_TO_A ((1 << m), dst, src1, src2, L_INPUT);
  dit (dst, m);
}

void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::dif_xni (u32 *arr,
                                                                      u32 n)
{
  MUL_B_TO_A ((1 << n), arr, ws0.get () + (1 << (n + 1)), L_INPUT);
  dif (arr, n);
}

void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::dit_xni (u32 *arr,
                                                                      u32 n)
{
  dit (arr, n);
  MUL_B_TO_A ((1 << n), arr, ws1.get () + (1 << (n + 1)), L_INPUT);
}

void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    internal_inv_faster (u32 *src, u32 *dst, u32 *tmp, u32 *tmp2, u32 *tmp3,
                         u32 len)
{ //9E(n)

  if (len == 1)
    {
      dst[0] = _fastpow (src[0], P - 2);
      return;
    }

  // 求出 f 的低 n/2 位的逆元 g_0, 存在 dst 的低 n/2 位中
  internal_inv_faster (src, dst, tmp, tmp2, tmp3, len >> 1);

  std::memcpy (tmp, src, sizeof (u32) * (len >> 1));
  std::memcpy (tmp2, dst, sizeof (u32) * (len >> 1));

  ADD_B_TIMES_C_TO_A (len >> 1, tmp, src + (len >> 1), ws0[3], L_INPUT);

  // 求出 f 在 mod x^(n/2)-i 循环卷积意义下的点值, 存在 tmp 的低 n/2 位中
  dif_xni (tmp, __builtin_ctz (len >> 1));

  // 求出 g_0 在 mod x^(n/2)-i 循环卷积意义下的点值, 存在 tmp2 的低 n/2 位中
  dif_xni (tmp2, __builtin_ctz (len >> 1));

  MUL_B_TIMES_B_TO_A (len >> 1, tmp, tmp2, L_INPUT);

  // 设 fg_0^2-g_0 = a * x^(n/2) + b * x^n + c * x^(3n/2)
  // tmp 中存的是 f*g_0*g_0 在 mod x^(n/2)-i 循环卷积意义下的结果，第 j 项是 i * a_j - b_j - i * c_j + g_0_j
  dit_xni (tmp, __builtin_ctz (len >> 1));

  std::memcpy (tmp2, src, sizeof (u32) * len);
  std::memcpy (tmp3, dst, sizeof (u32) * (len >> 1));
  std::memset (tmp3 + (len >> 1), 0, sizeof (u32) * (len >> 1));

  // 求出 f 在 mod x^n-1 循环卷积意义下的点值, 存在 tmp2 的 n 位中
  dif (tmp2, __builtin_ctz (len));

  // 求出 g_0 在 mod x^n-1 循环卷积意义下的点值, 存在 tmp3 的 n 位中
  dif (tmp3, __builtin_ctz (len));

  MUL_B_TIMES_B_TO_A (len, tmp2, tmp3, L_INPUT);

  // tmp2 中存的是 f*g_0*g_0 在 mod x^n-1 循环卷积意义下的结果
  // 对于低 n/2 项, 第 j 项是 b_j + g_0_j
  // 对于高 n/2 项, 第 j + n/2 项是 a_j + c_j
  dit (tmp2, __builtin_ctz (len));

// 合并答案
#if USE_AVX2
  if (len <= 8)
    {
      u32 mip = ws0[3], iv2 = _inv[1];

      for (u32 i = 0; i < (len >> 1); ++i)
        dst[i + (len >> 1)]
            = li.mul (li.sub (li.mul (li.sub (li.add (tmp[i], tmp2[i]),
                                              li.add (dst[i], dst[i])),
                                      mip),
                              tmp2[i + (len >> 1)]),
                      iv2);
    }
  else
    {
      __m256i *p1 = (__m256i *)tmp, *p2 = (__m256i *)(tmp2 + (len >> 1)),
              *p3 = (__m256i *)(tmp2), *p4 = (__m256i *)(dst + (len >> 1)),
              *p5 = (__m256i *)(dst);

      __m256i mip = _mm256_set1_epi32 (ws0[3]),
              iv2 = _mm256_set1_epi32 (_inv[1]);

      for (u32 i = 0; i < (len >> 1); i += 8, ++p1, ++p2, ++p3, ++p4, ++p5)
        (*p4) = la.mul (la.sub (la.mul (la.sub (la.add ((*p1), (*p3)),
                                                la.add ((*p5), (*p5))),
                                        mip),
                                (*p2)),
                        iv2);
    }
#else
  {
    u32 mip = ws0[3], iv2 = _inv[1];

    for (u32 i = 0; i < (len >> 1); ++i)
      dst[i + (len >> 1)]
          = li.mul (li.sub (li.mul (li.sub (li.add (tmp[i], tmp2[i]),
                                            li.add (dst[i], dst[i])),
                                    mip),
                            tmp2[i + (len >> 1)]),
                    iv2);
  }
#endif
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::internal_div (
    u32 *f, u32 *h, u32 *dst, u32 *tmp1, u32 *tmp2, u32 *tmp3, u32 len)
{ //check poly_div.md for more implementation detail

  if (len == 1)
    {
      dst[0] = li.mul (_fastpow (f[0], P - 2), h[0]);
      return;
    }

  if (len == 2)
    {
      u32 c  = _fastpow (f[0], P - 2);
      dst[0] = li.mul (c, h[0]);
      u32 d  = li.mul (li.neg (li.mul (c, c)), f[1]);
      dst[1] = li.add (li.mul (d, h[0]), li.mul (c, h[1]));
      return;
    }

  internal_inv_faster (f, dst, tmp1, tmp2, tmp3, (len >> 1));

  dit (tmp3, __builtin_ctz (len >> 1));

  dif (tmp3, __builtin_ctz (len >> 1));

  std::memcpy (tmp3 + (len >> 1), dst + (len >> 2), sizeof (u32) * (len >> 2));
  std::memset (tmp3 + (len >> 2) * 3, 0, sizeof (u32) * (len >> 2));

  dif (tmp3 + (len >> 1), __builtin_ctz (len >> 1));

  std::memcpy (tmp2, h, sizeof (u32) * (len >> 2));
  std::memcpy (tmp2 + (len >> 1), h + (len >> 2), sizeof (u32) * (len >> 2));
  std::memset (tmp2 + (len >> 2), 0, sizeof (u32) * (len >> 2));
  std::memset (tmp2 + (len >> 2) * 3, 0, sizeof (u32) * (len >> 2));

  dif (tmp2, __builtin_ctz (len >> 1));
  dif (tmp2 + (len >> 1), __builtin_ctz (len >> 1));

  trick_mul ((len >> 1), tmp3, tmp3 + (len >> 1), tmp2, tmp2 + (len >> 1),
             tmp1, tmp1 + (len >> 1));

  std::memset (tmp1 + (len >> 1), 0, sizeof (u32) * (len >> 1));
  std::memcpy (dst, tmp1, sizeof (u32) * (len >> 1));

  dif (tmp1, __builtin_ctz (len));

  std::memcpy (tmp2, f, sizeof (u32) * (len));

  dif (tmp2, __builtin_ctz (len));

  MUL_B_TO_A (len, tmp2, tmp1, L_INPUT);

  dit (tmp2, __builtin_ctz (len));

  SUB_B_FROM_A ((len >> 1), tmp2 + (len >> 1), h + (len >> 1), L_INPUT);

  std::memcpy (tmp2, tmp2 + (len >> 1), sizeof (u32) * (len >> 2));
  std::memcpy (tmp2 + (len >> 1), tmp2 + (len >> 2) * 3,
               sizeof (u32) * (len >> 2));
  std::memset (tmp2 + (len >> 2), 0, sizeof (u32) * (len >> 2));
  std::memset (tmp2 + (len >> 2) * 3, 0, sizeof (u32) * (len >> 2));

  dif (tmp2, __builtin_ctz (len >> 1));
  dif (tmp2 + (len >> 1), __builtin_ctz (len >> 1));

  trick_mul ((len >> 1), tmp3, tmp3 + (len >> 1), tmp2, tmp2 + (len >> 1),
             tmp1, tmp1 + (len >> 1));

  NEG_B_TO_A ((len >> 1), dst + (len >> 1), tmp1, L_INPUT);
}

void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::release ()
{
  ws0.reset ();
  ws1.reset ();
  _inv.reset ();
  num.reset ();
  fn = fb = mx = 0;
  for (u32 i = 0; i < tmp_size; ++i)
    tt[i].reset ();
}
u32
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::_fastpow (u32 a,
                                                                       u32 b)
{
  u32 ans = li.v (1), off = a;
  while (b)
    {
      if (b & 1)
        ans = li.mul (ans, off);
      off = li.mul (off, off);
      b >>= 1;
    }
  return ans;
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::init (
    u32 max_conv_size, u32 P0, u32 G0)
{
  if (P0 >= (1u << 30) || !P0)
    throw std::runtime_error ("invalid prime!");
  if (G0 >= P0 || (!G0))
    throw std::runtime_error ("invalid primitive root!");
  {
    if (!factorization::miller_rabin_u32 (P0))
      throw std::runtime_error ("invalid prime!");
    auto fact = factorization::pollard_rho_factorize_u32 (P0 - 1);
    for (auto &&v : fact)
      if (basic::fast_pow (G0, (P0 - 1) / v.first, P0) == 1)
        throw std::runtime_error ("invalid primitive root!");
  }
  if (max_conv_size >= (1u << 30))
    throw std::runtime_error ("invalid range!");
  max_conv_size = std::max (max_conv_size, 16u);
  li            = lmi (P0);
#if USE_AVX2
  la = lma (P0);
#endif
#if USE_AVX512
  l5 = lm5 (P0);
#endif
  release ();
  P = P0, G = G0;
  mx = max_conv_size;
  fn = 1;
  fb = 0;
  while (fn < (max_conv_size << 1))
    fn <<= 1, ++fb;
  if ((P0 - 1) % fn)
    throw std::runtime_error ("invalid range!");
  _inv = create_aligned_array<u32, 64> (fn + 32);
  ws0  = create_aligned_array<u32, 64> (fn + 32);
  ws1  = create_aligned_array<u32, 64> (fn + 32);
  num  = create_aligned_array<u32, 64> (fn + 32);
  for (u32 i = 0; i < tmp_size; ++i)
    tt[i] = create_aligned_array<u32, 64> (fn + 32);
  _inv[0] = li.v (1);
  for (u32 i = 2; i <= fn + 32; ++i)
    _inv[i - 1] = li.mul (li.v (P - P / i), _inv[(P % i) - 1]);
  for (u32 i = 1; i <= fn + 32; ++i)
    num[i - 1] = li.v (i);
  u32 j0 = _fastpow (li.v (G), (P - 1) / fn),
      j1 = _fastpow (_fastpow (li.v (G), (P - 2)), (P - 1) / fn);
  for (u32 mid = (fn >> 1); mid >= 1;
       mid >>= 1, j0 = li.mul (j0, j0), j1 = li.mul (j1, j1))
    {
      u32 w0 = li.v (1), w1 = li.v (1);
      for (u32 i = 0; i < mid; ++i, w0 = li.mul (w0, j0), w1 = li.mul (w1, j1))
        ws0[i + mid] = w0, ws1[i + mid] = w1;
    }
}
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    polynomial_kernel_ntt (const polynomial_kernel_ntt &d)
{
  fn = d.fn, fb = d.fb;
  P = d.P, G = d.G;
  mx = d.mx;
  li = d.li;
#if USE_AVX2
  la = d.la;
#endif
#if USE_AVX512
  l5 = d.l5;
#endif
  if (d.mx)
    {
      _inv = create_aligned_array<u32, 64> (fn + 32);
      ws0  = create_aligned_array<u32, 64> (fn + 32);
      ws1  = create_aligned_array<u32, 64> (fn + 32);
      num  = create_aligned_array<u32, 64> (fn + 32);
      for (u32 i = 0; i < tmp_size; ++i)
        tt[i] = create_aligned_array<u32, 64> (fn + 32);
      std::memcpy (ws0.get (), d.ws0.get (), sizeof (u32) * (fn + 32));
      std::memcpy (ws1.get (), d.ws1.get (), sizeof (u32) * (fn + 32));
      std::memcpy (_inv.get (), d._inv.get (), sizeof (u32) * (fn + 32));
      std::memcpy (num.get (), d.num.get (), sizeof (u32) * (fn + 32));
    }
}
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    polynomial_kernel_ntt (polynomial_kernel_ntt &&d)
{
  fn = d.fn, fb = d.fb;
  P = d.P, G = d.G;
  mx = d.mx;
  li = d.li;
#if USE_AVX2
  la = d.la;
#endif
#if USE_AVX512
  l5 = d.l5;
#endif
  fn = fb = d.mx = 0;
  _inv           = std::move (d._inv);
  ws0            = std::move (d.ws0);
  ws1            = std::move (d.ws1);
  num            = std::move (d.num);
  for (u32 i = 0; i < tmp_size; ++i)
    tt[i] = std::move (d.tt[i]);
}
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    polynomial_kernel_ntt ()
{
  fn = fb = mx = 0;
}
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    polynomial_kernel_ntt (u32 max_conv_size, u32 P0, u32 G0)
{
  init (max_conv_size, P0, G0);
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::dif (u32 *p,
                                                                  u32 n)
{
#if USE_AVX512
  u32 len = (1 << n);
  u32 *ws = ws0.get ();
  if (len < 16)
    {
      u32 t1, t2;
      for (u32 l = len; l >= 2; l >>= 1)
        for (u32 j = 0, mid = (l >> 1); j < len; j += l)
          {
            u32 *p1 = p + j, *p2 = p + j + mid, *ww = ws + mid;
            for (u32 i = 0; i < mid; ++i, ++p1, ++p2, ++ww)
              t1 = *p1, t2 = *p2, *p1 = li.add (t1, t2),
              *p2 = li.mul (li.sub (t1, t2), (*ww));
          }
    }
  else if (len <= (1 << NTT_partition_size))
    {
      __m512i *pp = (__m512i *)p, *p1, *p2, *ww;
      __m512i msk, val;
      __mmask16 smsk;
      for (u32 l = len; l > 16; l >>= 1)
        {
          u32 mid = (l >> 1);
          for (u32 j = 0; j < len; j += l)
            {
              p1 = (__m512i *)(p + j), p2 = (__m512i *)(p + j + mid),
              ww = (__m512i *)(ws + mid);
              for (u32 i = 0; i < mid; i += 16, ++p1, ++p2, ++ww)
                {
                  __m512i x = *p1, y = *p2;
                  *p1 = l5.add (x, y);
                  *p2 = l5.mul (l5.sub (x, y), *ww);
                }
            }
        }
      val = _mm512_setr_epi32 (ws[8], ws[8], ws[8], ws[8], ws[8], ws[8], ws[8],
                               ws[8], ws[8], ws[9], ws[10], ws[11], ws[12],
                               ws[13], ws[14], ws[15]);
      msk = _mm512_setr_epi32 (0, 0, 0, 0, 0, 0, 0, 0, P * 2, P * 2, P * 2,
                               P * 2, P * 2, P * 2, P * 2, P * 2);
      smsk = 0xff00;
      pp   = (__m512i *)p;
      for (u32 j = 0; j < len; j += 16, ++pp)
        {
          __m512i x = _mm512_shuffle_i64x2 (*pp, *pp, _MM_PERM_BADC);
          __m512i y = _mm512_mask_sub_epi32 (*pp, smsk, msk, *pp);
          *pp       = l5.mul (l5.add (x, y), val);
        }
      val = _mm512_setr_epi32 (ws[4], ws[4], ws[4], ws[4], ws[4], ws[5], ws[6],
                               ws[7], ws[4], ws[4], ws[4], ws[4], ws[4], ws[5],
                               ws[6], ws[7]);
      smsk = 0xf0f0;
      msk = _mm512_setr_epi32 (0, 0, 0, 0, P * 2, P * 2, P * 2, P * 2, 0, 0, 0,
                               0, P * 2, P * 2, P * 2, P * 2);
      pp  = (__m512i *)p;
      for (u32 j = 0; j < len; j += 16, ++pp)
        {
          __m512i x = _mm512_shuffle_i64x2 (*pp, *pp, _MM_PERM_CDAB);
          __m512i y = _mm512_mask_sub_epi32 (*pp, smsk, msk, *pp);
          *pp       = l5.mul (l5.add (x, y), val);
        }
      val = _mm512_setr_epi32 (ws[2], ws[2], ws[2], ws[3], ws[2], ws[2], ws[2],
                               ws[3], ws[2], ws[2], ws[2], ws[3], ws[2], ws[2],
                               ws[2], ws[3]);
      msk = _mm512_setr_epi32 (0, 0, P * 2, P * 2, 0, 0, P * 2, P * 2, 0, 0,
                               P * 2, P * 2, 0, 0, P * 2, P * 2);
      pp  = (__m512i *)p;
      smsk = 0xcccc;
      for (u32 j = 0; j < len; j += 16, ++pp)
        {
          __m512i x = _mm512_shuffle_epi32 (*pp, _MM_PERM_BADC);
          __m512i y = _mm512_mask_sub_epi32 (*pp, smsk, msk, *pp);
          *pp       = l5.mul (l5.add (x, y), val);
        }
      msk  = _mm512_setr_epi32 (0, P * 2, 0, P * 2, 0, P * 2, 0, P * 2, 0,
                               P * 2, 0, P * 2, 0, P * 2, 0, P * 2);
      pp   = (__m512i *)p;
      smsk = 0xaaaa;
      for (u32 j = 0; j < len; j += 16, ++pp)
        {
          __m512i x = _mm512_shuffle_epi32 (*pp, _MM_PERM_CDAB);
          __m512i y = _mm512_mask_sub_epi32 (*pp, smsk, msk, *pp);
          *pp       = l5.add (x, y);
        }
    }
  else
    {
      __m512i *p1 = (__m512i *)(p), *p2 = (__m512i *)(p + (len >> 2)),
              *p3 = (__m512i *)(p + (len >> 1)),
              *p4 = (__m512i *)(p + (len >> 2) * 3),
              *w1 = (__m512i *)(ws0.get () + (len >> 1)),
              *w2 = (__m512i *)(ws0.get () + (len >> 1) + (len >> 2)),
              *w3 = (__m512i *)(ws0.get () + (len >> 2));
      for (u32 i = 0; i < (len >> 2);
           i += 16, ++p1, ++p2, ++p3, ++p4, ++w2, ++w3, ++w1)
        {
          __m512i x = (*(p1)), y = (*(p2)), z = (*(p3)), w = (*(p4));
          __m512i r = l5.add (x, z), s = l5.mul (l5.sub (x, z), *w1);
          __m512i t = l5.add (y, w), q = l5.mul (l5.sub (y, w), *w2);
          (*(p1)) = l5.add (r, t);
          (*(p2)) = l5.mul (l5.sub (r, t), *w3);
          (*(p3)) = l5.add (s, q);
          (*(p4)) = l5.mul (l5.sub (s, q), *w3);
        }
      dif (p, n - 2);
      dif (p + (1 << (n - 2)), n - 2);
      dif (p + (1 << (n - 1)), n - 2);
      dif (p + (1 << (n - 2)) * 3, n - 2);
    }
#elif USE_AVX2
  u32 len = (1 << n);
  u32 *ws = ws0.get ();
  if (len < 8)
    {
      u32 t1, t2;
      for (u32 l = len; l >= 2; l >>= 1)
        for (u32 j = 0, mid = (l >> 1); j < len; j += l)
          {
            u32 *p1 = p + j, *p2 = p + j + mid, *ww = ws + mid;
            for (u32 i = 0; i < mid; ++i, ++p1, ++p2, ++ww)
              t1 = *p1, t2 = *p2, *p1 = li.add (t1, t2),
              *p2 = li.mul (li.sub (t1, t2), (*ww));
          }
    }
  else if (len <= (1 << NTT_partition_size))
    {
      __m256i *pp = (__m256i *)p, *p1, *p2, *ww;
      __m256i msk, val;
      for (u32 l = len; l > 8; l >>= 1)
        {
          u32 mid = (l >> 1);
          for (u32 j = 0; j < len; j += l)
            {
              p1 = (__m256i *)(p + j), p2 = (__m256i *)(p + j + mid),
              ww = (__m256i *)(ws + mid);
              for (u32 i = 0; i < mid; i += 8, ++p1, ++p2, ++ww)
                {
                  __m256i x = *p1, y = *p2;
                  *p1 = la.add (x, y);
                  *p2 = la.mul (la.sub (x, y), *ww);
                }
            }
        }
      val = _mm256_setr_epi32 (ws[4], ws[4], ws[4], ws[4], ws[4], ws[5], ws[6],
                               ws[7]);
      msk = _mm256_setr_epi32 (0, 0, 0, 0, P * 2, P * 2, P * 2, P * 2);
      pp  = (__m256i *)p;
      for (u32 j = 0; j < len; j += 8, ++pp)
        {
          __m256i x = _mm256_permute4x64_epi64 (*pp, 0x4E);
          __m256i y = _mm256_add_epi32 (
              _mm256_sign_epi32 (
                  *pp, _mm256_setr_epi32 (1, 1, 1, 1, -1, -1, -1, -1)),
              msk);
          *pp = la.mul (la.add (x, y), val);
        }
      val = _mm256_setr_epi32 (ws[2], ws[2], ws[2], ws[3], ws[2], ws[2], ws[2],
                               ws[3]);
      msk = _mm256_setr_epi32 (0, 0, P * 2, P * 2, 0, 0, P * 2, P * 2);
      pp  = (__m256i *)p;
      for (u32 j = 0; j < len; j += 8, ++pp)
        {
          __m256i x = _mm256_shuffle_epi32 (*pp, 0x4E);
          __m256i y = _mm256_add_epi32 (
              _mm256_sign_epi32 (
                  *pp, _mm256_setr_epi32 (1, 1, -1, -1, 1, 1, -1, -1)),
              msk);
          *pp = la.mul (la.add (x, y), val);
        }
      msk = _mm256_setr_epi32 (0, P * 2, 0, P * 2, 0, P * 2, 0, P * 2);
      pp  = (__m256i *)p;
      for (u32 j = 0; j < len; j += 8, ++pp)
        {
          __m256i x = _mm256_shuffle_epi32 (*pp, 0xB1);
          __m256i y = _mm256_add_epi32 (
              _mm256_sign_epi32 (
                  *pp, _mm256_setr_epi32 (1, -1, 1, -1, 1, -1, 1, -1)),
              msk);
          *pp = la.add (x, y);
        }
    }
  else
    {
      __m256i *p1 = (__m256i *)(p), *p2 = (__m256i *)(p + (len >> 2)),
              *p3 = (__m256i *)(p + (len >> 1)),
              *p4 = (__m256i *)(p + (len >> 2) * 3),
              *w1 = (__m256i *)(ws0.get () + (len >> 1)),
              *w2 = (__m256i *)(ws0.get () + (len >> 1) + (len >> 2)),
              *w3 = (__m256i *)(ws0.get () + (len >> 2));
      for (u32 i = 0; i < (len >> 2);
           i += 8, ++p1, ++p2, ++p3, ++p4, ++w2, ++w3, ++w1)
        {
          __m256i x = (*(p1)), y = (*(p2)), z = (*(p3)), w = (*(p4));
          __m256i r = la.add (x, z), s = la.mul (la.sub (x, z), *w1);
          __m256i t = la.add (y, w), q = la.mul (la.sub (y, w), *w2);
          (*(p1)) = la.add (r, t);
          (*(p2)) = la.mul (la.sub (r, t), *w3);
          (*(p3)) = la.add (s, q);
          (*(p4)) = la.mul (la.sub (s, q), *w3);
        }
      dif (p, n - 2);
      dif (p + (1 << (n - 2)), n - 2);
      dif (p + (1 << (n - 1)), n - 2);
      dif (p + (1 << (n - 2)) * 3, n - 2);
    }
#else
  u32 len = (1 << n);
  u32 t1, t2;
  u32 *ws = ws0.get ();
  for (u32 l = len; l >= 2; l >>= 1)
    for (u32 j = 0, mid = (l >> 1); j < len; j += l)
      {
        u32 *p1 = p + j, *p2 = p + j + mid, *ww = ws + mid;
        for (u32 i = 0; i < mid; ++i, ++p1, ++p2, ++ww)
          t1 = *p1, t2 = *p2, *p1 = li.add (t1, t2),
          *p2 = li.mul (li.sub (t1, t2), (*ww));
      }
#endif
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::dit (
    u32 *p, u32 n, bool inverse_coef)
{
#if USE_AVX512
  u32 len = (1 << n);
  u32 *ws = ws1.get ();
  if (len < 16)
    {
      u32 t1, t2;
      for (u32 l = 2; l <= len; l <<= 1)
        for (u32 j = 0, mid = (l >> 1); j < len; j += l)
          {
            u32 *p1 = p + j, *p2 = p + j + mid, *ww = ws + mid;
            for (u32 i = 0; i < mid; ++i, ++p1, ++p2, ++ww)
              t1 = *p1, t2 = li.mul ((*p2), (*ww)), *p1 = li.add (t1, t2),
              *p2 = li.sub (t1, t2);
          }
      u32 co  = _inv[len - 1];
      u32 *p1 = p;
      for (u32 i = 0; i < len; ++i, ++p1)
        (*p1) = li.mul (co, (*p1));
    }
  else if (len <= (1 << NTT_partition_size))
    {
      __m512i *pp = (__m512i *)p, *p1, *p2, *ww;
      __m512i msk, val;
      __mmask16 smsk;
      msk  = _mm512_setr_epi32 (0, P * 2, 0, P * 2, 0, P * 2, 0, P * 2, 0,
                               P * 2, 0, P * 2, 0, P * 2, 0, P * 2);
      smsk = 0xaaaa;
      pp   = (__m512i *)p;
      for (u32 j = 0; j < len; j += 16, ++pp)
        {
          __m512i x = _mm512_shuffle_epi32 (*pp, _MM_PERM_CDAB);
          __m512i y = _mm512_mask_sub_epi32 (*pp, smsk, msk, *pp);
          *pp       = l5.add (x, y);
        }
      val = _mm512_setr_epi32 (ws[2], ws[3], li.neg (ws[2]), li.neg (ws[3]),
                               ws[2], ws[3], li.neg (ws[2]), li.neg (ws[3]),
                               ws[2], ws[3], li.neg (ws[2]), li.neg (ws[3]),
                               ws[2], ws[3], li.neg (ws[2]), li.neg (ws[3]));
      pp  = (__m512i *)p;
      for (u32 j = 0; j < len; j += 16, ++pp)
        {
          __m512i x = _mm512_shuffle_epi32 (*pp, _MM_PERM_BABA);
          __m512i y = _mm512_shuffle_epi32 (*pp, _MM_PERM_DCDC);
          *pp       = l5.add (x, l5.mul (y, val));
        }
      val = _mm512_setr_epi32 (ws[4], ws[5], ws[6], ws[7], li.neg (ws[4]),
                               li.neg (ws[5]), li.neg (ws[6]), li.neg (ws[7]),
                               ws[4], ws[5], ws[6], ws[7], li.neg (ws[4]),
                               li.neg (ws[5]), li.neg (ws[6]), li.neg (ws[7]));
      pp  = (__m512i *)p;
      for (u32 j = 0; j < len; j += 16, ++pp)
        {
          __m512i x = _mm512_shuffle_i64x2 (*pp, *pp, _MM_PERM_CCAA);
          __m512i y = _mm512_shuffle_i64x2 (*pp, *pp, _MM_PERM_DDBB);
          *pp       = l5.add (x, l5.mul (y, val));
        }
      val = _mm512_setr_epi32 (
          ws[8], ws[9], ws[10], ws[11], ws[12], ws[13], ws[14], ws[15],
          li.neg (ws[8]), li.neg (ws[9]), li.neg (ws[10]), li.neg (ws[11]),
          li.neg (ws[12]), li.neg (ws[13]), li.neg (ws[14]), li.neg (ws[15]));
      pp = (__m512i *)p;
      for (u32 j = 0; j < len; j += 16, ++pp)
        {
          __m512i x = _mm512_shuffle_i64x2 (*pp, *pp, _MM_PERM_BABA);
          __m512i y = _mm512_shuffle_i64x2 (*pp, *pp, _MM_PERM_DCDC);
          *pp       = l5.add (x, l5.mul (y, val));
        }
      for (u32 l = 32; l <= len; l <<= 1)
        {
          u32 mid = (l >> 1);
          for (u32 j = 0; j < len; j += l)
            {
              p1 = (__m512i *)(p + j), p2 = (__m512i *)(p + j + mid),
              ww = (__m512i *)(ws + mid);
              for (u32 i = 0; i < mid; i += 16, ++p1, ++p2, ++ww)
                {
                  __m512i x = *p1, y = l5.mul (*p2, *ww);
                  *p1 = l5.add (x, y);
                  *p2 = l5.sub (x, y);
                }
            }
        }
      if (inverse_coef)
        {
          __m512i co = _mm512_set1_epi32 (_inv[len - 1]);
          pp         = (__m512i *)p;
          for (u32 i = 0; i < len; i += 16, ++pp)
            (*pp) = l5.mul (*pp, co);
        }
    }
  else
    {
      dit (p, n - 2, false);
      dit (p + (1 << (n - 2)), n - 2, false);
      dit (p + (1 << (n - 1)), n - 2, false);
      dit (p + (1 << (n - 2)) * 3, n - 2, false);
      __m512i *p1 = (__m512i *)(p), *p2 = (__m512i *)(p + (len >> 2)),
              *p3 = (__m512i *)(p + (len >> 1)),
              *p4 = (__m512i *)(p + (len >> 2) * 3),
              *w1 = (__m512i *)(ws + (len >> 1)),
              *w2 = (__m512i *)(ws + (len >> 1) + (len >> 2)),
              *w3 = (__m512i *)(ws + (len >> 2));
      for (u32 i = 0; i < (len >> 2);
           i += 16, ++p1, ++p2, ++p3, ++p4, ++w2, ++w3, ++w1)
        {
          __m512i x = (*(p1)), y = (*(p2)), z = (*(p3)), w = (*(p4));
          __m512i h = l5.mul (y, *w3), k = l5.mul (w, *w3);
          __m512i t = l5.mul (l5.add (z, k), *w1),
                  q = l5.mul (l5.sub (z, k), *w2);
          __m512i r = l5.add (x, h), s = l5.sub (x, h);
          (*(p1)) = l5.add (r, t);
          (*(p2)) = l5.add (s, q);
          (*(p3)) = l5.sub (r, t);
          (*(p4)) = l5.sub (s, q);
        }
      if (inverse_coef)
        {
          __m512i co = _mm512_set1_epi32 (_inv[len - 1]);
          p1         = (__m512i *)p;
          for (u32 i = 0; i < len; i += 16, ++p1)
            (*p1) = l5.mul (*p1, co);
        }
    }
#elif USE_AVX2
  u32 len = (1 << n);
  u32 *ws = ws1.get ();
  if (len < 8)
    {
      u32 t1, t2;
      for (u32 l = 2; l <= len; l <<= 1)
        for (u32 j = 0, mid = (l >> 1); j < len; j += l)
          {
            u32 *p1 = p + j, *p2 = p + j + mid, *ww = ws + mid;
            for (u32 i = 0; i < mid; ++i, ++p1, ++p2, ++ww)
              t1 = *p1, t2 = li.mul ((*p2), (*ww)), *p1 = li.add (t1, t2),
              *p2 = li.sub (t1, t2);
          }
      u32 co  = _inv[len - 1];
      u32 *p1 = p;
      for (u32 i = 0; i < len; ++i, ++p1)
        (*p1) = li.mul (co, (*p1));
    }
  else if (len <= (1 << NTT_partition_size))
    {
      __m256i *pp = (__m256i *)p, *p1, *p2, *ww;
      __m256i msk, val;
      msk = _mm256_setr_epi32 (0, P * 2, 0, P * 2, 0, P * 2, 0, P * 2);
      pp  = (__m256i *)p;
      for (u32 j = 0; j < len; j += 8, ++pp)
        {
          __m256i x = _mm256_shuffle_epi32 (*pp, 0xB1);
          __m256i y = _mm256_add_epi32 (
              _mm256_sign_epi32 (
                  *pp, _mm256_setr_epi32 (1, -1, 1, -1, 1, -1, 1, -1)),
              msk);
          *pp = la.add (x, y);
        }
      val = _mm256_setr_epi32 (ws[2], ws[3], li.neg (ws[2]), li.neg (ws[3]),
                               ws[2], ws[3], li.neg (ws[2]), li.neg (ws[3]));
      pp  = (__m256i *)p;
      for (u32 j = 0; j < len; j += 8, ++pp)
        {
          __m256i x = _mm256_shuffle_epi32 (*pp, 0x44);
          __m256i y = _mm256_shuffle_epi32 (*pp, 0xEE);
          *pp       = la.add (x, la.mul (y, val));
        }
      val = _mm256_setr_epi32 (ws[4], ws[5], ws[6], ws[7], li.neg (ws[4]),
                               li.neg (ws[5]), li.neg (ws[6]), li.neg (ws[7]));
      pp  = (__m256i *)p;
      for (u32 j = 0; j < len; j += 8, ++pp)
        {
          __m256i x = _mm256_permute4x64_epi64 (*pp, 0x44);
          __m256i y = _mm256_permute4x64_epi64 (*pp, 0xEE);
          *pp       = la.add (x, la.mul (y, val));
        }
      for (u32 l = 16; l <= len; l <<= 1)
        {
          u32 mid = (l >> 1);
          for (u32 j = 0; j < len; j += l)
            {
              p1 = (__m256i *)(p + j), p2 = (__m256i *)(p + j + mid),
              ww = (__m256i *)(ws + mid);
              for (u32 i = 0; i < mid; i += 8, ++p1, ++p2, ++ww)
                {
                  __m256i x = *p1, y = la.mul (*p2, *ww);
                  *p1 = la.add (x, y);
                  *p2 = la.sub (x, y);
                }
            }
        }
      if (inverse_coef)
        {
          __m256i co = _mm256_set1_epi32 (_inv[len - 1]);
          pp         = (__m256i *)p;
          for (u32 i = 0; i < len; i += 8, ++pp)
            (*pp) = la.mul (*pp, co);
        }
    }
  else
    {
      dit (p, n - 2, false);
      dit (p + (1 << (n - 2)), n - 2, false);
      dit (p + (1 << (n - 1)), n - 2, false);
      dit (p + (1 << (n - 2)) * 3, n - 2, false);
      __m256i *p1 = (__m256i *)(p), *p2 = (__m256i *)(p + (len >> 2)),
              *p3 = (__m256i *)(p + (len >> 1)),
              *p4 = (__m256i *)(p + (len >> 2) * 3),
              *w1 = (__m256i *)(ws + (len >> 1)),
              *w2 = (__m256i *)(ws + (len >> 1) + (len >> 2)),
              *w3 = (__m256i *)(ws + (len >> 2));
      for (u32 i = 0; i < (len >> 2);
           i += 8, ++p1, ++p2, ++p3, ++p4, ++w2, ++w3, ++w1)
        {
          __m256i x = (*(p1)), y = (*(p2)), z = (*(p3)), w = (*(p4));
          __m256i h = la.mul (y, *w3), k = la.mul (w, *w3);
          __m256i t = la.mul (la.add (z, k), *w1),
                  q = la.mul (la.sub (z, k), *w2);
          __m256i r = la.add (x, h), s = la.sub (x, h);
          (*(p1)) = la.add (r, t);
          (*(p2)) = la.add (s, q);
          (*(p3)) = la.sub (r, t);
          (*(p4)) = la.sub (s, q);
        }
      if (inverse_coef)
        {
          __m256i co = _mm256_set1_epi32 (_inv[len - 1]);
          p1         = (__m256i *)p;
          for (u32 i = 0; i < len; i += 8, ++p1)
            (*p1) = la.mul (*p1, co);
        }
    }
#else
  u32 len = (1 << n);
  u32 t1, t2;
  u32 *ws = ws1.get ();
  for (u32 l = 2; l <= len; l <<= 1)
    for (u32 j = 0, mid = (l >> 1); j < len; j += l)
      {
        u32 *p1 = p + j, *p2 = p + j + mid, *ww = ws + mid;
        for (u32 i = 0; i < mid; ++i, ++p1, ++p2, ++ww)
          t1 = *p1, t2 = li.mul ((*p2), (*ww)), *p1 = li.add (t1, t2),
          *p2 = li.sub (t1, t2);
      }
  u32 co  = _inv[len - 1];
  u32 *p1 = p;
  for (u32 i = 0; i < len; ++i, ++p1)
    (*p1) = li.mul (co, (*p1));
#endif
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    internal_transpose_mul (u32 *src1, u32 *src2, u32 *dst, u32 m)
{
  std::reverse (src1, src1 + (1 << m));
  internal_mul (src1, src2, dst, m);
  std::reverse (dst, dst + (1 << m));
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::mul (
    const power_series_ring::poly &a, const power_series_ring::poly &b)
{
  u32 la = a.size (), lb = b.size ();
  if ((!la) && (!lb))
    return poly ();
  if (la > mx || lb > mx)
    throw std::runtime_error ("Convolution size out of range!");
  u32 m = 0;
  if (la + lb > 2)
    m = 32 - __builtin_clz (la + lb - 2);
  std::memcpy (tt[0].get (), &a[0], sizeof (u32) * la);
  std::memset (tt[0].get () + la, 0, sizeof (u32) * ((1 << m) - la));
  std::memcpy (tt[1].get (), &b[0], sizeof (u32) * lb);
  std::memset (tt[1].get () + lb, 0, sizeof (u32) * ((1 << m) - lb));
  internal_mul (tt[0].get (), tt[1].get (), tt[2].get (), m);
  poly ret (la + lb - 1);
  std::memcpy (&ret[0], tt[2].get (), sizeof (u32) * (la + lb - 1));
  return ret;
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::transpose_mul (
    const power_series_ring::poly &a, const power_series_ring::poly &b)
{
  u32 la = a.size (), lb = b.size ();
  if ((!la) && (!lb))
    return poly ();
  if (la > mx || lb > mx)
    throw std::runtime_error ("Convolution size out of range!");
  u32 m = 0;
  if (la + lb > 2)
    m = 32 - __builtin_clz (la + lb - 2);
  std::memcpy (tt[0].get (), &a[0], sizeof (u32) * la);
  std::memset (tt[0].get () + la, 0, sizeof (u32) * ((1 << m) - la));
  std::memcpy (tt[1].get (), &b[0], sizeof (u32) * lb);
  std::memset (tt[1].get () + lb, 0, sizeof (u32) * ((1 << m) - lb));
  internal_transpose_mul (tt[0].get (), tt[1].get (), tt[2].get (), m);
  poly ret (la);
  std::memcpy (&ret[0], tt[2].get (), sizeof (u32) * (la));
  return ret;
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::internal_inv (
    u32 *src, u32 *dst, u32 *tmp, u32 *tmp2, u32 len)
{ //10E(n) x^n->x^{2n}
  if (len == 1)
    {
      dst[0] = _fastpow (src[0], P - 2);
      return;
    }
  internal_inv (src, dst, tmp, tmp2, len >> 1);
  std::memcpy (tmp, src, sizeof (u32) * len);
  std::memcpy (tmp2, dst, sizeof (u32) * (len >> 1));
  std::memset (tmp2 + (len >> 1), 0, sizeof (u32) * (len >> 1));
  std::memset (dst + (len >> 1), 0, sizeof (u32) * (len >> 1));
  dif (tmp, __builtin_ctz (len));
  dif (tmp2, __builtin_ctz (len));
#if USE_AVX2
  if (len <= 4)
    {
      for (u32 i = 0; i < len; ++i)
        tmp[i] = li.mul (tmp[i], tmp2[i]);
    }
  else
    {
      __m256i *p1 = (__m256i *)tmp2, *p2 = (__m256i *)tmp;
      for (u32 i = 0; i < len; i += 8, ++p1, ++p2)
        (*p2) = la.mul ((*p1), (*p2));
    }
#else
  for (u32 i = 0; i < len; ++i)
    tmp[i] = li.mul (tmp[i], tmp2[i]);
#endif
  dit (tmp, __builtin_ctz (len));
  std::memset (tmp, 0, sizeof (u32) * (len >> 1));
  dif (tmp, __builtin_ctz (len));
#if USE_AVX2
  if (len <= 4)
    {
      for (u32 i = 0; i < len; ++i)
        tmp[i] = li.mul (tmp[i], tmp2[i]);
    }
  else
    {
      __m256i *p1 = (__m256i *)tmp2, *p2 = (__m256i *)tmp;
      for (u32 i = 0; i < len; i += 8, ++p1, ++p2)
        (*p2) = la.mul ((*p1), (*p2));
    }
#else
  for (u32 i = 0; i < len; ++i)
    tmp[i] = li.mul (tmp[i], tmp2[i]);
#endif
  dit (tmp, __builtin_ctz (len));
#if USE_AVX2
  if (len <= 8)
    {
      for (u32 i = (len >> 1); i < len; ++i)
        dst[i] = li.neg (tmp[i]);
    }
  else
    {
      __m256i *p1 = (__m256i *)(tmp + (len >> 1)),
              *p2 = (__m256i *)(dst + (len >> 1));
      for (u32 i = 0; i < (len >> 1); i += 8, ++p1, ++p2)
        (*p2) = la.sub (_mm256_setzero_si256 (), (*p1));
    }
#else
  for (u32 i = (len >> 1); i < len; ++i)
    dst[i] = li.neg (tmp[i]);
#endif
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::inv (
    const power_series_ring::poly &src)
{
  u32 la = src.size ();
  if (!la)
    throw std::runtime_error ("Inversion calculation of empty polynomial!");
  if ((la * 4) > fn)
    throw std::runtime_error ("Inversion calculation size out of range!");
  if (!li.rv (src[0].get_val ()))
    {
      throw std::runtime_error ("Inversion calculation of polynomial which "
                                "has constant not equal to 1!");
    }
  u32 m = 0;
  if (la > 1)
    m = 32 - __builtin_clz (la - 1);
  std::memcpy (tt[0].get (), &src[0], sizeof (u32) * la);
  std::memset (tt[0].get () + la, 0, sizeof (u32) * ((1 << m) - la));
  // internal_inv(tt[0].get(),tt[1].get(),tt[2].get(),tt[3].get(),(1<<m));
  internal_inv_faster (tt[0].get (), tt[1].get (), tt[2].get (), tt[3].get (),
                       tt[4].get (), (1 << m));
  poly ret (la);
  std::memcpy (&ret[0], tt[1].get (), sizeof (u32) * la);
  return ret;
}
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    ~polynomial_kernel_ntt ()
{
  release ();
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::internal_ln (
    u32 *src, u32 *dst, u32 *tmp1, u32 *tmp2, u32 *tmp3, u32 len)
{
#if USE_AVX2
  u32 pos     = 1;
  __m256i *pp = (__m256i *)tmp1, *iv = (__m256i *)num.get ();
  u32 *p1 = src + 1;
  for (; pos + 8 <= len; pos += 8, p1 += 8, ++pp, ++iv)
    *pp = la.mul (_mm256_loadu_si256 ((__m256i *)p1), *iv);
  for (; pos < len; ++pos)
    tmp1[pos - 1] = li.mul (src[pos], num[pos - 1]);
  tmp1[len - 1] = li.v (0);
#else
  u32 *p1 = src + 1, *p2 = tmp1, *p3 = num.get ();
  for (u32 i = 1; i < len; ++i, ++p1, ++p2, ++p3)
    *p2 = li.mul ((*p1), (*p3));
  tmp1[len - 1] = li.v (0);
#endif
  internal_inv (src, dst, tmp2, tmp3, len);
  std::memset (dst + len, 0, sizeof (u32) * len);
  std::memset (tmp1 + len, 0, sizeof (u32) * len);
  internal_mul (tmp1, dst, tmp2, __builtin_ctz (len << 1));
#if USE_AVX2
  u32 ps       = 1;
  __m256i *pp0 = (__m256i *)tmp2, *iv0 = (__m256i *)_inv.get ();
  u32 *p10 = dst + 1;
  for (; ps + 8 <= len; ps += 8, p10 += 8, ++pp0, ++iv0)
    _mm256_storeu_si256 ((__m256i *)p10, la.mul (*pp0, *iv0));
  dst[0] = li.v (0);
  for (; ps < len; ++ps)
    dst[ps] = li.mul (_inv[ps - 1], tmp2[ps - 1]);
#else
  dst[0]        = li.v (0);
  u32 *p10 = dst + 1, *p20 = tmp2, *p30 = _inv.get ();
  for (u32 i = 1; i < len; ++i, ++p10, ++p20, ++p30)
    *p10 = li.mul ((*p20), (*p30));
#endif
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    internal_ln_faster (u32 *src, u32 *dst, u32 *tmp1, u32 *tmp2, u32 *tmp3,
                        u32 *tmp4, u32 len)
{ //12E(n)
#if USE_AVX2
  u32 pos     = 1;
  __m256i *pp = (__m256i *)tmp1, *iv = (__m256i *)num.get ();
  u32 *p1 = src + 1;
  for (; pos + 8 <= len; pos += 8, p1 += 8, ++pp, ++iv)
    *pp = la.mul (_mm256_loadu_si256 ((__m256i *)p1), *iv);
  for (; pos < len; ++pos)
    tmp1[pos - 1] = li.mul (src[pos], num[pos - 1]);
  tmp1[len - 1] = li.v (0);
#else
  u32 *p1 = src + 1, *p2 = tmp1, *p3 = num.get ();
  for (u32 i = 1; i < len; ++i, ++p1, ++p2, ++p3)
    *p2 = li.mul ((*p1), (*p3));
  tmp1[len - 1] = li.v (0);
#endif
  internal_div (src, tmp1, tmp2, dst, tmp3, tmp4, len);
#if USE_AVX2
  u32 ps       = 1;
  __m256i *pp0 = (__m256i *)tmp2, *iv0 = (__m256i *)_inv.get ();
  u32 *p10 = dst + 1;
  for (; ps + 8 <= len; ps += 8, p10 += 8, ++pp0, ++iv0)
    _mm256_storeu_si256 ((__m256i *)p10, la.mul (*pp0, *iv0));
  dst[0] = li.v (0);
  for (; ps < len; ++ps)
    dst[ps] = li.mul (_inv[ps - 1], tmp2[ps - 1]);
#else
  dst[0]        = li.v (0);
  u32 *p10 = dst + 1, *p20 = tmp2, *p30 = _inv.get ();
  for (u32 i = 1; i < len; ++i, ++p10, ++p20, ++p30)
    *p10 = li.mul ((*p20), (*p30));
#endif
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::ln (
    const poly &src)
{
  u32 la = src.size ();
  if (!la)
    throw std::runtime_error ("Ln calculation of empty polynomial!");
  if ((la * 2) > fn)
    throw std::runtime_error ("Ln calculation size out of range!");
  if (li.rv (src[0].get_val ()) != 1)
    {
      throw std::runtime_error (
          "Ln calculation of polynomial which has constant not equal to 1!");
    }
  u32 m = 0;
  if (la > 1)
    m = 32 - __builtin_clz (la - 1);
  std::memcpy (tt[0].get (), &src[0], sizeof (u32) * la);
  std::memset (tt[0].get () + la, 0, sizeof (u32) * ((1 << m) - la));
  // internal_ln(tt[0].get(),tt[1].get(),tt[2].get(),tt[3].get(),tt[4].get(),(1<<m));
  internal_ln_faster (tt[0].get (), tt[1].get (), tt[2].get (), tt[3].get (),
                      tt[4].get (), tt[5].get (), (1 << m));
  poly ret (la);
  std::memcpy (&ret[0], tt[1].get (), sizeof (u32) * la);
  return ret;
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::internal_exp (
    u32 *src, u32 *dst, u32 *gn, u32 *gxni, u32 *h, u32 *tmp1, u32 *tmp2,
    u32 *tmp3, u32 len, bool calc_h)
{
  if (len == 1)
    {
      dst[0] = li.v (1);
      return;
    }
  else if (len == 2)
    {
      dst[0] = li.v (1);
      dst[1] = src[1];
      gn[0] = li.add (dst[0], dst[1]), gn[1] = li.sub (dst[0], dst[1]);
      gxni[0] = li.add (li.mul (dst[1], ws0[3]), dst[0]);
      h[0]    = li.v (1);
      h[1]    = li.neg (dst[1]);
      return;
    }
  internal_exp (src, dst, gn, gxni, h, tmp1, tmp2, tmp3, (len >> 1), true);
#if USE_AVX2
  {
    u32 pos     = 1;
    __m256i *pp = (__m256i *)tmp1, *iv = (__m256i *)num.get ();
    u32 *p1 = src + 1;
    for (; pos + 8 <= (len >> 1); pos += 8, p1 += 8, ++pp, ++iv)
      *pp = la.mul (_mm256_loadu_si256 ((__m256i *)p1), *iv);
    for (; pos < (len >> 1); ++pos)
      tmp1[pos - 1] = li.mul (src[pos], num[pos - 1]);
    tmp1[(len >> 1) - 1] = li.v (0);
  }
#else
  {
    u32 *p1 = src + 1, *p2 = tmp1, *p3 = num.get ();
    for (u32 i = 1; i < (len >> 1); ++i, ++p1, ++p2, ++p3)
      *p2 = li.mul ((*p1), (*p3));
    tmp1[(len >> 1) - 1] = li.v (0);
  }
#endif
  dif (tmp1, __builtin_ctz (len >> 1));
#if USE_AVX2
  if (len < 8)
    {
      for (u32 i = 0; i < (len >> 1); ++i)
        tmp1[i] = li.mul (tmp1[i], gn[i]);
    }
  else
    {
      __m256i *p1 = (__m256i *)(tmp1), *p2 = (__m256i *)(gn);
      for (u32 i = 0; i < (len >> 1); i += 8, ++p1, ++p2)
        (*p1) = la.mul ((*p1), (*p2));
    }
#else
  for (u32 i = 0; i < (len >> 1); ++i)
    tmp1[i] = li.mul (tmp1[i], gn[i]);
#endif
  dit (tmp1, __builtin_ctz (len >> 1));
#if USE_AVX2
  {
    u32 pos     = 1;
    __m256i *pp = (__m256i *)tmp1, *iv = (__m256i *)num.get ();
    u32 *p1 = dst + 1;
    for (; pos + 8 <= (len >> 1); pos += 8, p1 += 8, ++pp, ++iv)
      *pp = la.sub (la.mul (_mm256_loadu_si256 ((__m256i *)p1), *iv), (*pp));
    for (; pos < (len >> 1); ++pos)
      tmp1[pos - 1] = li.sub (li.mul (dst[pos], num[pos - 1]), tmp1[pos - 1]);
    tmp1[(len >> 1) - 1] = li.neg (tmp1[(len >> 1) - 1]);
  }
#else
  {
    u32 *p1 = dst + 1, *p2 = tmp1, *p3 = num.get ();
    for (u32 i = 1; i < (len >> 1); ++i, ++p1, ++p2, ++p3)
      *p2 = li.sub (li.mul ((*p1), (*p3)), (*p2));
    tmp1[(len >> 1) - 1] = li.neg (tmp1[(len >> 1) - 1]);
  }
#endif
  std::memmove (tmp1 + 1, tmp1, sizeof (u32) * (len >> 1));
  tmp1[0] = tmp1[(len >> 1)];
  std::memset (tmp1 + (len >> 1), 0, sizeof (u32) * (len >> 1));
  dif (tmp1, __builtin_ctz (len));
  std::memcpy (tmp3, h, sizeof (u32) * (len >> 1));
  std::memset (tmp3 + (len >> 1), 0, sizeof (u32) * (len >> 1));
  dif (tmp3, __builtin_ctz (len));
#if USE_AVX2
  if (len < 8)
    {
      for (u32 i = 0; i < len; ++i)
        tmp1[i] = li.mul (tmp3[i], tmp1[i]);
    }
  else
    {
      __m256i *p1 = (__m256i *)tmp1, *p2 = (__m256i *)tmp3;
      __m256i tt;
      for (u32 i = 0; i < len; i += 8, ++p1, ++p2)
        (*p1) = la.mul ((*p1), (*p2));
    }
#else
  for (u32 i = 0; i < len; ++i)
    tmp1[i] = li.mul (tmp3[i], tmp1[i]);
#endif
  dit (tmp1, __builtin_ctz (len));
#if USE_AVX2
  if (len <= 8)
    {
      for (u32 i = 0; i < (len >> 1); ++i)
        tmp2[i] = li.sub (src[i + (len >> 1)],
                          li.mul (_inv[i + (len >> 1) - 1], tmp1[i]));
    }
  else
    {
      __m256i *p1 = (__m256i *)tmp1, *p3 = (__m256i *)(tmp2),
              *p4 = (__m256i *)(src + (len >> 1));
      u32 *p2     = _inv.get () + (len >> 1) - 1;
      for (u32 i = 0; i < (len >> 1); i += 8, ++p1, p2 += 8, ++p3, ++p4)
        (*p3) = la.sub ((*p4),
                        la.mul ((*p1), _mm256_loadu_si256 ((__m256i *)p2)));
    }
#else
  {
    for (u32 i = 0; i < (len >> 1); ++i)
      tmp2[i] = li.sub (src[i + (len >> 1)],
                        li.mul (_inv[i + (len >> 1) - 1], tmp1[i]));
  }
#endif
  std::memset (tmp2 + (len >> 1), 0, sizeof (u32) * (len >> 1));
  dif (tmp2, __builtin_ctz (len));
#if USE_AVX2
  if (len <= 16)
    {
      u32 mip = ws1[3];
      for (u32 i = 0; i < (len >> 2); ++i)
        tmp1[i] = li.mul (
            li.mul (li.add (dst[i], li.mul (dst[i + (len >> 2)], mip)),
                    ws0[(len >> 1) + i]),
            ws0[(len >> 2) + i]);
    }
  else
    {
      __m256i *p1 = (__m256i *)dst, *p2 = (__m256i *)(dst + (len >> 2)),
              *p3 = (__m256i *)(tmp1),
              *p4 = (__m256i *)(ws0.get () + (len >> 1)),
              *p5 = (__m256i *)(ws0.get () + (len >> 2));
      __m256i mip = _mm256_set1_epi32 (ws1[3]);
      for (u32 i = 0; i < (len >> 2); i += 8, ++p1, ++p2, ++p3, ++p4, ++p5)
        (*p3) = la.mul (la.add ((*p1), la.mul ((*p2), mip)),
                        la.mul ((*p4), (*p5)));
    }
#else
  {
    u32 mip = ws1[3];
    for (u32 i = 0; i < (len >> 2); ++i)
      tmp1[i]
          = li.mul (li.mul (li.add (dst[i], li.mul (dst[i + (len >> 2)], mip)),
                            ws0[(len >> 1) + i]),
                    ws0[(len >> 2) + i]);
  }
#endif
  dif (tmp1, __builtin_ctz (len >> 2));
  std::memcpy (tmp1 + (len >> 2) * 3, tmp1, sizeof (u32) * (len >> 2));
  std::memcpy (tmp1, gn, sizeof (u32) * (len >> 1));
  std::memcpy (tmp1 + (len >> 1), gxni, sizeof (u32) * (len >> 2));
#if USE_AVX2
  if (len <= 4)
    {
      for (u32 i = 0; i < len; ++i)
        tmp1[i] = li.mul (tmp2[i], tmp1[i]);
    }
  else
    {
      __m256i *p1 = (__m256i *)tmp1, *p2 = (__m256i *)(tmp2);
      for (u32 i = 0; i < len; i += 8, ++p1, ++p2)
        (*p1) = la.mul ((*p1), (*p2));
    }
#else
  for (u32 i = 0; i < len; ++i)
    tmp1[i] = li.mul (tmp2[i], tmp1[i]);
#endif
  dit (tmp1, __builtin_ctz (len));
  std::memcpy (dst + (len >> 1), tmp1, sizeof (u32) * (len >> 1));
  //inv iteration start
  if (!calc_h)
    return;
  std::memcpy (gxni, dst, sizeof (u32) * (len >> 1));
  std::memcpy (tmp2, h, sizeof (u32) * (len >> 1));
#if USE_AVX2
  if (len <= 8)
    {
      u32 mip = ws0[3];
      for (u32 i = 0; i < (len >> 1); ++i)
        gxni[i] = li.add (gxni[i], li.mul (mip, dst[i + (len >> 1)]));
    }
  else
    {
      __m256i mip = _mm256_set1_epi32 (ws0[3]);
      __m256i *p1 = (__m256i *)(dst + (len >> 1)), *p2 = (__m256i *)gxni;
      for (u32 i = 0; i < (len >> 1); i += 8, ++p1, ++p2)
        (*p2) = la.add ((*p2), la.mul ((*p1), mip));
    }
#else
  {
    u32 mip = ws0[3];
    for (u32 i = 0; i < (len >> 1); ++i)
      gxni[i] = li.add (gxni[i], li.mul (mip, dst[i + (len >> 1)]));
  }
#endif
  dif_xni (gxni, __builtin_ctz (len >> 1));
  dif_xni (tmp2, __builtin_ctz (len >> 1));
#if USE_AVX2
  if (len <= 8)
    {
      for (u32 i = 0; i < (len >> 1); ++i)
        tmp2[i] = li.mul (li.mul (tmp2[i], gxni[i]), tmp2[i]);
    }
  else
    {
      __m256i *p1 = (__m256i *)tmp2, *p2 = (__m256i *)gxni;
      for (u32 i = 0; i < (len >> 1); i += 8, ++p1, ++p2)
        (*p1) = la.mul ((*p2), la.mul ((*p1), (*p1)));
    }
#else
  for (u32 i = 0; i < (len >> 1); ++i)
    tmp2[i] = li.mul (li.mul (tmp2[i], gxni[i]), tmp2[i]);
#endif
  dit_xni (tmp2, __builtin_ctz (len >> 1));
  std::memcpy (gn, dst, sizeof (u32) * len);
  dif (gn, __builtin_ctz (len));
#if USE_AVX2
  if (len <= 8)
    {
      for (u32 i = 0; i < len; ++i)
        tmp3[i] = li.mul (li.mul (tmp3[i], gn[i]), tmp3[i]);
    }
  else
    {
      __m256i *p1 = (__m256i *)tmp3, *p2 = (__m256i *)gn;
      for (u32 i = 0; i < len; i += 8, ++p1, ++p2)
        (*p1) = la.mul ((*p2), la.mul ((*p1), (*p1)));
    }
#else
  for (u32 i = 0; i < len; ++i)
    tmp3[i] = li.mul (li.mul (tmp3[i], gn[i]), tmp3[i]);
#endif
  dit (tmp3, __builtin_ctz (len));
#if USE_AVX2
  if (len <= 8)
    {
      u32 mip = ws0[3], iv2 = _inv[1];
      for (u32 i = 0; i < (len >> 1); ++i)
        h[i + (len >> 1)]
            = li.mul (li.sub (li.mul (li.sub (li.add (tmp2[i], tmp3[i]),
                                              li.add (h[i], h[i])),
                                      mip),
                              tmp3[i + (len >> 1)]),
                      iv2);
    }
  else
    {
      __m256i *p1 = (__m256i *)tmp2, *p2 = (__m256i *)(tmp3 + (len >> 1)),
              *p3 = (__m256i *)(tmp3), *p4 = (__m256i *)(h + (len >> 1)),
              *p5 = (__m256i *)(h);
      __m256i mip = _mm256_set1_epi32 (ws0[3]),
              iv2 = _mm256_set1_epi32 (_inv[1]);
      for (u32 i = 0; i < (len >> 1); i += 8, ++p1, ++p2, ++p3, ++p4, ++p5)
        (*p4) = la.mul (la.sub (la.mul (la.sub (la.add ((*p1), (*p3)),
                                                la.add ((*p5), (*p5))),
                                        mip),
                                (*p2)),
                        iv2);
    }
#else
  {
    u32 mip = ws0[3], iv2 = _inv[1];
    for (u32 i = 0; i < (len >> 1); ++i)
      h[i + (len >> 1)]
          = li.mul (li.sub (li.mul (li.sub (li.add (tmp2[i], tmp3[i]),
                                            li.add (h[i], h[i])),
                                    mip),
                            tmp3[i + (len >> 1)]),
                    iv2);
  }
#endif
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::exp (
    const poly &src)
{
  u32 la = src.size ();
  if (!la)
    throw std::runtime_error ("Exp calculation of empty polynomial!");
  if ((la * 2) > fn)
    throw std::runtime_error ("Exp calculation size out of range!");
  if (li.rv (src[0].get_val ()) != 0)
    {
      throw std::runtime_error (
          "Exp calculation of polynomial which has constant not equal to 0!");
    }
  u32 m = 0;
  if (la > 1)
    m = 32 - __builtin_clz (la - 1);
  std::memcpy (tt[0].get (), &src[0], sizeof (u32) * la);
  std::memset (tt[0].get () + la, 0, sizeof (u32) * ((1 << m) - la));
  internal_exp (tt[0].get (), tt[1].get (), tt[2].get (), tt[3].get (),
                tt[4].get (), tt[5].get (), tt[6].get (), tt[7].get (),
                (1 << m));
  poly ret (la);
  std::memcpy (&ret[0], tt[1].get (), sizeof (u32) * la);
  return ret;
}
std::array<long long, 9>
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::test (u32 T)
{
  std::mt19937 rnd (default_mod);
  std::uniform_int_distribution<u32> rng{ 0, default_mod - 1 };
  u32 len = (fn >> 2);
  for (u32 i = 0; i < len; ++i)
    tt[0][i] = tt[tmp_size - 1][i] = li.v (rng (rnd));
  auto dif_start = std::chrono::system_clock::now ();
  for (u32 i = 0; i < T; ++i)
    dif (tt[0].get (), __builtin_ctz (len));
  auto dif_end   = std::chrono::system_clock::now ();
  auto dit_start = std::chrono::system_clock::now ();
  for (u32 i = 0; i < T; ++i)
    dit (tt[0].get (), __builtin_ctz (len));
  auto dit_end = std::chrono::system_clock::now ();
  for (u32 i = 0; i < len; ++i)
    assert (li.reds (tt[0][i]) == li.reds (tt[tmp_size - 1][i]));
  tt[0][0]            = li.v (1);
  tt[tmp_size - 1][0] = li.v (1);
  auto inv_start      = std::chrono::system_clock::now ();
  for (u32 i = 0; i < T; ++i)
    internal_inv (tt[i & 1].get (), tt[i & 1 ^ 1].get (), tt[2].get (),
                  tt[3].get (), len);
  auto inv_end = std::chrono::system_clock::now ();
  for (u32 i = 0; i < len; ++i)
    assert (li.reds (tt[0][i]) == li.reds (tt[tmp_size - 1][i]));
  auto inv_faster_start = std::chrono::system_clock::now ();
  for (u32 i = 0; i < T; ++i)
    internal_inv_faster (tt[i & 1].get (), tt[i & 1 ^ 1].get (), tt[2].get (),
                         tt[3].get (), tt[4].get (), len);
  auto inv_faster_end = std::chrono::system_clock::now ();
  for (u32 i = 0; i < len; ++i)
    assert (li.reds (tt[0][i]) == li.reds (tt[tmp_size - 1][i]));
  auto ln_start = std::chrono::system_clock::now ();
  for (u32 i = 0; i < T; ++i)
    tt[0][0] = li.v (1), internal_ln (tt[0].get (), tt[1].get (), tt[2].get (),
                                      tt[3].get (), tt[4].get (), len);
  auto ln_end          = std::chrono::system_clock::now ();
  auto ln_faster_start = std::chrono::system_clock::now ();
  for (u32 i = 0; i < T; ++i)
    tt[i & 1][0] = li.v (1),
           internal_ln_faster (tt[i & 1].get (), tt[i & 1 ^ 1].get (),
                               tt[2].get (), tt[3].get (), tt[4].get (),
                               tt[5].get (), len);
  auto ln_faster_end = std::chrono::system_clock::now ();
  auto exp_start     = std::chrono::system_clock::now ();
  for (u32 i = 0; i < T; ++i)
    tt[i & 1][0] = li.v (0),
           internal_exp (tt[i & 1].get (), tt[i & 1 ^ 1].get (), tt[2].get (),
                         tt[3].get (), tt[4].get (), tt[5].get (),
                         tt[6].get (), tt[7].get (), len);
  auto exp_end = std::chrono::system_clock::now ();
  for (u32 i = 0; i < len; ++i)
    assert (li.reds (tt[0][i]) == li.reds (tt[tmp_size - 1][i]));
  poly src1 (len), src2 (len), dst;
  std::uniform_int_distribution<u32> stp{ 1, 100 };
  for (u32 i = 0, cur = 0; i < len; ++i)
    cur += stp (rnd), src1[i] = ui2mi (tt[0][i]), src2[i] = li.v (cur);
  auto mei_start = std::chrono::system_clock::now ();
  for (u32 i = 0; i < T; ++i)
    dst = multipoint_eval_interpolation (src1, src2);
  auto mei_end = std::chrono::system_clock::now ();
  std::vector<std::pair<mi, mi> > src3 (len);
  for (u32 i = 0; i < len; ++i)
    src3[i] = std::make_pair (src2[i], dst[i]);
  auto li_start = std::chrono::system_clock::now ();
  for (u32 i = 0; i < T; ++i)
    src1 = lagrange_interpolation (src3);
  for (u32 i = 0; i < len; ++i)
    assert (li.reds (src1[i].get_val ()) == li.reds (tt[tmp_size - 1][i]));
  auto li_end       = std::chrono::system_clock::now ();
  auto dif_duration = std::chrono::duration_cast<std::chrono::microseconds> (
           dif_end - dif_start),
       dit_duration = std::chrono::duration_cast<std::chrono::microseconds> (
           dit_end - dit_start),
       inv_duration = std::chrono::duration_cast<std::chrono::microseconds> (
           inv_end - inv_start),
       inv_faster_duration
       = std::chrono::duration_cast<std::chrono::microseconds> (
           inv_faster_end - inv_faster_start),
       ln_duration = std::chrono::duration_cast<std::chrono::microseconds> (
           ln_end - ln_start),
       ln_faster_duration
       = std::chrono::duration_cast<std::chrono::microseconds> (
           ln_faster_end - ln_faster_start),
       exp_duration = std::chrono::duration_cast<std::chrono::microseconds> (
           exp_end - exp_start),
       mei_duration = std::chrono::duration_cast<std::chrono::microseconds> (
           mei_end - mei_start),
       li_duration = std::chrono::duration_cast<std::chrono::microseconds> (
           li_end - li_start);
  return { dif_duration.count (), dit_duration.count (),
           inv_duration.count (), inv_faster_duration.count (),
           ln_duration.count (),  ln_faster_duration.count (),
           exp_duration.count (), mei_duration.count (),
           li_duration.count () };
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    internal_multipoint_eval_interpolation_calc_Q (
        std::vector<poly> &Q_storage, const poly &input_coef, u32 l, u32 r,
        u32 id)
{
  if (l == r)
    {
      Q_storage[id]
          = { ui2mi (li.v (1)), ui2mi (li.neg (input_coef[l].get_val ())) };
      return;
    }
  u32 mid = (l + r) >> 1;
  internal_multipoint_eval_interpolation_calc_Q (Q_storage, input_coef, l, mid,
                                                 id << 1);
  internal_multipoint_eval_interpolation_calc_Q (Q_storage, input_coef,
                                                 mid + 1, r, id << 1 | 1);
  Q_storage[id] = mul (Q_storage[id << 1], Q_storage[id << 1 | 1]);
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    internal_multipoint_eval_interpolation_calc_P (
        const std::vector<poly> &Q_storage, std::vector<poly> &P_stack,
        poly &result_coef, u32 l, u32 r, u32 id, u32 dep)
{
  if (l == r)
    {
      result_coef[l] = P_stack[dep][0];
      return;
    }
  P_stack[dep].resize (r - l + 1);
  u32 mid          = (l + r) >> 1;
  P_stack[dep + 1] = transpose_mul (P_stack[dep], Q_storage[id << 1 | 1]);
  internal_multipoint_eval_interpolation_calc_P (
      Q_storage, P_stack, result_coef, l, mid, id << 1, dep + 1);
  P_stack[dep + 1] = transpose_mul (P_stack[dep], Q_storage[id << 1]);
  internal_multipoint_eval_interpolation_calc_P (
      Q_storage, P_stack, result_coef, mid + 1, r, id << 1 | 1, dep + 1);
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    multipoint_eval_interpolation (const poly &a, const poly &b)
{
  u32 point_count = b.size (), poly_count = a.size (),
      maxnm = std::max (point_count, poly_count);
  if (!poly_count)
    throw std::runtime_error (
        "Multipoint eval interpolation of empty polynomial!");
  if (maxnm * 4 > fn)
    throw std::runtime_error (
        "Multipoint eval interpolation size out of range!");
  if (!point_count)
    return {};
  std::vector<poly> Q_storage (point_count << 2),
      P_stack ((32 - __builtin_clz (point_count)) * 2);
  internal_multipoint_eval_interpolation_calc_Q (Q_storage, b, 0,
                                                 point_count - 1, 1);
  poly ans (point_count);
  Q_storage[1].resize (maxnm);
  P_stack[0] = transpose_mul (a, inv (Q_storage[1]));
  internal_multipoint_eval_interpolation_calc_P (Q_storage, P_stack, ans, 0,
                                                 point_count - 1, 1, 0);
  return ans;
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    internal_lagrange_interpolation_dvc_mul (
        u32 l, u32 r, const poly &a, u32 id,
        std::vector<std::pair<poly, poly> > &R_storage)
{
  if (l == r)
    {
      R_storage[id].first
          = { ui2mi (li.neg (a[l].get_val ())), ui2mi (li.v (1)) };
      return;
    }
  u32 mid = (l + r) >> 1;
  internal_lagrange_interpolation_dvc_mul (l, mid, a, id << 1, R_storage);
  internal_lagrange_interpolation_dvc_mul (mid + 1, r, a, id << 1 | 1,
                                           R_storage);
  u32 m = 32 - __builtin_clz (r - l + 1);
  R_storage[id].first
      = mul (R_storage[id << 1].first, R_storage[id << 1 | 1].first);
  R_storage[id << 1].second.resize ((1 << m));
  R_storage[id << 1 | 1].second.resize ((1 << m));
  std::memcpy (&R_storage[id << 1].second[0], tt[0].get (),
               sizeof (mi) * (1 << m));
  std::memcpy (&R_storage[id << 1 | 1].second[0], tt[1].get (),
               sizeof (mi) * (1 << m));
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::derivative (
    const poly &a)
{
  if (!a.size ())
    throw std::runtime_error ("Derivative calculation of empty polynomial!");
  if (a.size () > fn)
    throw std::runtime_error ("Derivative calculation size out of range!");
  u32 len = a.size ();
  poly ret (len - 1);
  for (u32 i = 0; i < len - 1; ++i)
    ret[i] = ui2mi (li.mul (li.v (i + 1), a[i + 1].get_val ()));
  return ret;
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::integrate (
    const poly &a)
{
  if (!a.size ())
    throw std::runtime_error ("Integrate calculation of empty polynomial!");
  if (a.size () >= fn)
    throw std::runtime_error ("Integrate calculation size out of range!");
  u32 len = a.size ();
  poly ret (len + 1);
  ret[0] = ui2mi (li.v (0));
  for (u32 i = 1; i <= len; ++i)
    ret[i] = ui2mi (li.mul (_inv[i - 1], a[i - 1].get_val ()));
  return ret;
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::add (
    const poly &a, const poly &b)
{
  u32 la = a.size (), lb = b.size (), len = std::max (la, lb);
  poly ret (len);
  std::memcpy (&ret[0], &b[0], sizeof (mi) * (lb));
  for (u32 i = 0; i < la; ++i)
    ret[i] = ui2mi (li.add (ret[i].get_val (), a[i].get_val ()));
  return ret;
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::sub (
    const poly &a, const poly &b)
{
  u32 la = a.size (), lb = b.size (), len = std::max (la, lb);
  poly ret (len);
  std::memcpy (&ret[0], &a[0], sizeof (mi) * (la));
  for (u32 i = 0; i < lb; ++i)
    ret[i] = ui2mi (li.sub (ret[i].get_val (), b[i].get_val ()));
  return ret;
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    internal_lagrange_interpolation_calc_P (
        const std::vector<std::pair<poly, poly> > &R_storage,
        std::vector<poly> &P_stack, poly &result_coef, u32 l, u32 r, u32 id,
        u32 dep)
{
  if (l == r)
    {
      result_coef[l] = P_stack[dep][0];
      return;
    }
  P_stack[dep].resize (r - l + 1);
  u32 mid = (l + r) >> 1;
  P_stack[dep + 1]
      = transpose_mul (P_stack[dep], rev (R_storage[id << 1 | 1].first));
  internal_lagrange_interpolation_calc_P (R_storage, P_stack, result_coef, l,
                                          mid, id << 1, dep + 1);
  P_stack[dep + 1]
      = transpose_mul (P_stack[dep], rev (R_storage[id << 1].first));
  internal_lagrange_interpolation_calc_P (R_storage, P_stack, result_coef,
                                          mid + 1, r, id << 1 | 1, dep + 1);
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    internal_lagrange_interpolation_dvc_mul_ans (
        u32 l, u32 r, const poly &a, u32 id,
        const std::vector<std::pair<poly, poly> > &R_storage)
{
  if (l == r)
    return { a[l] };
  u32 mid   = (l + r) >> 1;
  auto &&lp = internal_lagrange_interpolation_dvc_mul_ans (l, mid, a, id << 1,
                                                           R_storage);
  auto &&rp = internal_lagrange_interpolation_dvc_mul_ans (
      mid + 1, r, a, id << 1 | 1, R_storage);
  u32 m = 32 - __builtin_clz (r - l + 1);
  std::memcpy (tt[0].get (), &lp[0], sizeof (mi) * (lp.size ()));
  std::memset (tt[0].get () + lp.size (), 0,
               sizeof (mi) * ((1 << m) - lp.size ()));
  std::memcpy (tt[1].get (), &rp[0], sizeof (mi) * (rp.size ()));
  std::memset (tt[1].get () + rp.size (), 0,
               sizeof (mi) * ((1 << m) - rp.size ()));
  dif (tt[0].get (), m);
  dif (tt[1].get (), m);
  for (u32 i = 0; i < (1 << m); ++i)
    tt[0][i] = li.add (
        li.mul (tt[0][i], R_storage[id << 1 | 1].second[i].get_val ()),
        li.mul (tt[1][i], R_storage[id << 1].second[i].get_val ()));
  dit (tt[0].get (), m);
  poly ret (r - l + 1);
  std::memcpy (&ret[0], tt[0].get (), sizeof (mi) * (r - l + 1));
  return ret;
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::rev (
    const poly &a)
{
  poly ret (a);
  std::reverse (ret.begin (), ret.end ());
  return ret;
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_ntt::
    lagrange_interpolation (const std::vector<std::pair<mi, mi> > &a)
{
  if (!a.size ())
    throw std::runtime_error ("Lagrange interpolation of zero coefficients!");
  auto d   = a;
  auto cmp = [] (const std::pair<mi, mi> &a, const std::pair<mi, mi> &b) {
    return a.first.get_val () < b.first.get_val ()
           || (a.first.get_val () == b.first.get_val ()
               && a.second.get_val () < b.second.get_val ());
  };
  auto ceq = [] (const std::pair<mi, mi> &a, const std::pair<mi, mi> &b) {
    return a.first.get_val () == b.first.get_val ()
           && a.second.get_val () == b.second.get_val ();
  };
  std::sort (d.begin (), d.end (), cmp);
  d.resize (std::unique (d.begin (), d.end (), ceq) - d.begin ());
  u32 len = d.size ();
  if (len * 4 > fn)
    throw std::runtime_error ("Lagrange interpolation size out of range!");
  for (u32 i = 1; i < len; ++i)
    if (d[i].first.get_val () == d[i - 1].first.get_val ())
      throw std::runtime_error (
          "Polynomial has multiple values on one position!");
  if (len == 1)
    {
      poly ret = poly ({ d[0].second });
      ret.resize (a.size ());
      return ret;
    }
  poly pts (len);
  for (u32 i = 0; i < len; ++i)
    pts[i] = d[i].first;
  std::vector<std::pair<poly, poly> > R_storage (len << 2);
  std::vector<poly> P_stack ((32 - __builtin_clz (len)) * 2);
  internal_lagrange_interpolation_dvc_mul (0, len - 1, pts, 1, R_storage);
  poly rf = rev (R_storage[1].first);
  rf.resize (len);
  P_stack[0] = transpose_mul (derivative (R_storage[1].first), inv (rf));
  poly coef (len);
  internal_lagrange_interpolation_calc_P (R_storage, P_stack, coef, 0, len - 1,
                                          1, 0);
  for (u32 i = 0; i < len; ++i)
    coef[i] = ui2mi (
        li.mul (_fastpow (coef[i].get_val (), P - 2), d[i].second.get_val ()));
  poly ret = internal_lagrange_interpolation_dvc_mul_ans (0, len - 1, coef, 1,
                                                          R_storage);
  ret.resize (a.size ());
  return ret;
}
power_series_ring::polynomial_kernel::polynomial_kernel_mtt::
    polynomial_kernel_mtt ()
    : F1 (P1), F2 (P2), F3 (P3), li1 (P1), li2 (P2), li3 (P3)
{
  P = fn = 0;
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_mtt::release ()
{
  k1.release ();
  k2.release ();
  k3.release ();
  P = fn = 0;
  _inv.reset ();
}
void
power_series_ring::polynomial_kernel::polynomial_kernel_mtt::init (
    u32 max_conv_size, u32 P0)
{
  if (P0 >= (1u << 30) || !P0)
    throw std::runtime_error ("invalid prime!");
  if (!factorization::miller_rabin_u32 (P0))
    throw std::runtime_error ("invalid prime!");
  try
    {
      release ();
      k1.init (max_conv_size, P1, G1);
      k2.init (max_conv_size, P2, G2);
      k3.init (max_conv_size, P3, G3);
      P  = P0;
      fn = k1.fn;
      li = lmi (P);
      F  = fast_mod_32 (P);
      if (P <= fn + 32)
        throw std::runtime_error ("invalid prime!");
      _inv    = create_aligned_array<u32, 64> (fn + 32);
      _inv[0] = li.v (1);
      for (u32 i = 2; i <= fn + 32; ++i)
        _inv[i - 1] = li.mul (li.v (P - P / i), _inv[(P % i) - 1]);
    }
  catch (std::exception &e)
    {
      P = 0;
      throw std::runtime_error (e.what ());
    }
}
power_series_ring::polynomial_kernel::polynomial_kernel_mtt::
    polynomial_kernel_mtt (u32 max_conv_size, u32 P0)
    : F1 (P1), F2 (P2), F3 (P3), li1 (P1), li2 (P2), li3 (P3)
{
  init (max_conv_size, P0);
}
power_series_ring::polynomial_kernel::polynomial_kernel_mtt::
    ~polynomial_kernel_mtt ()
{
  release ();
}
power_series_ring::polynomial_kernel::polynomial_kernel_mtt::
    polynomial_kernel_mtt (const polynomial_kernel_mtt &d)
    : P (d.P), fn (d.fn), k1 (d.k1), k2 (d.k2), k3 (d.k3), F1 (P1), F2 (P2),
      F3 (P3), F (d.F), li (d.li), li1 (P1), li2 (P2), li3 (P3)
{
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_mtt::add (
    const poly &a, const poly &b)
{
  u32 la = a.size (), lb = b.size (), len = std::max (la, lb);
  poly ret (len);
  std::memcpy (&ret[0], &b[0], sizeof (mi) * (lb));
  for (u32 i = 0; i < la; ++i)
    ret[i] = ui2mi (li.add (ret[i].get_val (), a[i].get_val ()));
  return ret;
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_mtt::sub (
    const poly &a, const poly &b)
{
  u32 la = a.size (), lb = b.size (), len = std::max (la, lb);
  poly ret (len);
  std::memcpy (&ret[0], &a[0], sizeof (mi) * (la));
  for (u32 i = 0; i < lb; ++i)
    ret[i] = ui2mi (li.sub (ret[i].get_val (), b[i].get_val ()));
  return ret;
}
power_series_ring::poly
power_series_ring::polynomial_kernel::polynomial_kernel_mtt::mul (
    const power_series_ring::poly &a, const power_series_ring::poly &b)
{
  u32 la = a.size (), lb = b.size ();
  if ((!la) && (!lb))
    return poly ();
  if ((la + lb - 1) > fn)
    throw std::runtime_error ("Convolution size out of range!");
  poly a1 (a.size ()), a2 (a.size ()), a3 (a.size ()), b1 (b.size ()),
      b2 (b.size ()), b3 (b.size ());
  for (u32 i = 0; i < a.size (); ++i)
    {
      u32 ra = li.rv (a[i].get_val ());
      a1[i]  = ui2mi (li1.v (ra));
      a2[i]  = ui2mi (li2.v (ra));
      a3[i]  = ui2mi (li3.v (ra));
    }
  for (u32 i = 0; i < b.size (); ++i)
    {
      u32 rb = li.rv (b[i].get_val ());
      b1[i]  = ui2mi (li1.v (rb));
      b2[i]  = ui2mi (li2.v (rb));
      b3[i]  = ui2mi (li3.v (rb));
    }
  poly r1 = k1.mul (a1, b1), r2 = k2.mul (a2, b2), r3 = k3.mul (a3, b3);
  poly ret (la + lb - 1);
  u32 I3 = F.reduce (1ull * P1 * P2);
  for (int i = 0; i < la + lb - 1; ++i)
    {
      u32 x1 = li1.rv (r1[i].get_val ());
      u32 x2 = li2.rv (r2[i].get_val ());
      u32 x3 = li3.rv (r3[i].get_val ());
      u32 k1 = F2.reduce (1ull * (x2 - x1 + P2) * I1);
      ull x4 = 1ull * k1 * P1 + x1;
      u32 k2 = F3.reduce ((x3 - F3.reduce (x4) + P3) * I2);
      u32 x  = F.reduce (x4 + 1ull * k2 * I3);
      ret[i] = ui2mi (li.v (x));
    }
  return ret;
}

#undef USE_AVX2

#undef USE_AVX512

#undef L_INPUT

#undef L_PARAM
}