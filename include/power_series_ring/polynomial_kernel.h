/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_POWER_SERIES_RING_POLYNOMIAL_KERNEL_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_POWER_SERIES_RING_POLYNOMIAL_KERNEL_H_

#include <array>
#include <modulo/modint.h>
#include <type/basic_typedef.h>
#include <type/type.h>
#include <vector>

namespace math
{
namespace power_series_ring
{
typedef std::vector<mi> poly;
namespace polynomial_kernel
{
class polynomial_kernel_mtt;
class polynomial_kernel_ntt
{
private:
  static constexpr ui tmp_size = 9;
  aligned_array<ui, 64> ws0, ws1, _inv, tt[tmp_size], num;
  ui P, G;
  ui fn, fb, mx;
  void release ();
  ui _fastpow (ui a, ui b);
  void dif (ui *arr, ui n);
  void dit (ui *arr, ui n, bool last_layer = true);
  void dif_xni (ui *arr, ui n);
  void dit_xni (ui *arr, ui n);
  void trick_mul (u32 _len, u32 *_pt1, u32 *_pt2, u32 *_pt3, u32 *_pt4,
                  u32 *_pt5, u32 *_pt6);
  void internal_mul (ui *src1, ui *src2, ui *dst, ui m);
  void internal_transpose_mul (ui *src1, ui *src2, ui *dst, ui m);
  void internal_div (u32 *f, u32 *h, u32 *dst, u32 *tmp1, u32 *tmp2, u32 *tmp3,
                     u32 len);
  void internal_inv (ui *src, ui *dst, ui *tmp, ui *tmp2, ui len);
  void internal_inv_faster (ui *src, ui *dst, ui *tmp, ui *tmp2, ui *tmp3,
                            ui len);
  void internal_ln (ui *src, ui *dst, ui *tmp1, ui *tmp2, ui *tmp3, ui len);
  void internal_ln_faster (ui *src, ui *dst, ui *tmp, ui *tmp2, ui *tmp3,
                           ui *tmp4, ui len);
  void internal_exp (ui *src, ui *dst, ui *gn, ui *gxni, ui *h, ui *tmp1,
                     ui *tmp2, ui *tmp3, ui len, bool calc_h = false);
  void internal_multipoint_eval_interpolation_calc_Q (
      std::vector<poly> &Q_storage, const poly &input_coef, ui l, ui r, ui id);
  void internal_multipoint_eval_interpolation_calc_P (
      const std::vector<poly> &Q_storage, std::vector<poly> &P_stack,
      poly &result_coef, ui l, ui r, ui id, ui dep);
  void internal_lagrange_interpolation_dvc_mul (
      ui l, ui r, const poly &a, ui id,
      std::vector<std::pair<poly, poly> > &R_storage);
  void internal_lagrange_interpolation_calc_P (
      const std::vector<std::pair<poly, poly> > &R_storage,
      std::vector<poly> &P_stack, poly &result_coef, ui l, ui r, ui id,
      ui dep);
  poly internal_lagrange_interpolation_dvc_mul_ans (
      ui l, ui r, const poly &a, ui id,
      const std::vector<std::pair<poly, poly> > &R_storage);
  lmi li;
#if defined(__AVX__) && defined(__AVX2__)
  lma la;
#endif
#if defined(__AVX512F__) && defined(__AVX512DQ__)
  lm5 l5;
#endif
public:
  friend class polynomial_kernel_mtt;
  polynomial_kernel_ntt (ui max_conv_size, ui P0, ui G0);
  void init (ui max_conv_size, ui P0, ui G0);
  polynomial_kernel_ntt (const polynomial_kernel_ntt &d);
  polynomial_kernel_ntt (polynomial_kernel_ntt &&d);
  polynomial_kernel_ntt ();
  ~polynomial_kernel_ntt ();
  poly rev (const poly &a);
  poly mul (const poly &a, const poly &b);
  poly transpose_mul (const poly &a, const poly &b);
  poly multipoint_eval_interpolation (const poly &a, const poly &b);
  poly lagrange_interpolation (const std::vector<std::pair<mi, mi> > &a);
  poly inv (const poly &src);
  poly ln (const poly &src);
  poly exp (const poly &src);
  poly derivative (const poly &src);
  poly integrate (const poly &src);
  poly add (const poly &src1, const poly &src2);
  poly sub (const poly &src1, const poly &src2);
  std::array<long long, 9> test (ui T);
};
class polynomial_kernel_mtt
{
private:
  static constexpr ui P1 = 167772161, G1 = 3, P2 = 469762049, G2 = 3,
                      P3 = 754974721, G3 = 11, I1 = 104391568, I2 = 190329765;
  polynomial_kernel_ntt k1, k2, k3;
  ui P, fn;
  lmi li, li1, li2, li3;
  fast_mod_32 F, F1, F2, F3;
  aligned_array<ui, 64> _inv;
  void release ();

public:
  void init (ui max_conv_size, ui P0);
  polynomial_kernel_mtt (ui max_conv_size, ui P0);
  polynomial_kernel_mtt (const polynomial_kernel_mtt &d);
  polynomial_kernel_mtt ();
  ~polynomial_kernel_mtt ();
  poly add (const poly &a, const poly &b);
  poly sub (const poly &a, const poly &b);
  poly mul (const poly &a, const poly &b);
};
}
}
}

#endif