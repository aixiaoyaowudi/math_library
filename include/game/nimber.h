/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_GAME_NIMBER_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_GAME_NIMBER_H_

#include <cstdint>
#include <game/constants/nimber_table.h>
#include <iostream>
#include <type/basic_typedef.h>

namespace math
{
namespace game
{
static inline ui __attribute__ ((__always_inline__))
__nimber_mul_4 (ui x, ui y)
{
  if (!x || !y)
    return 0;
  ui k = (ui)nimber_2_16_log_table[x] + (ui)nimber_2_16_log_table[y];
  return nimber_2_16_exp_table[(k & ((1 << 16) - 1)) + (k >> 16)];
}
struct nimber_int
{
  ui val;
  nimber_int () = default;
#define INLINE_OP __attribute__ ((__always_inline__))
  INLINE_OP
  nimber_int (ui k)
      : val (k)
  {
  }
  INLINE_OP nimber_int &
  operator= (const nimber_int &a)
  {
    val = a.val;
    return *this;
  }
  INLINE_OP nimber_int &
  operator+= (const nimber_int &a)
  {
    val ^= a.val;
    return *this;
  }
  INLINE_OP nimber_int &
  operator-= (const nimber_int &a)
  {
    val ^= a.val;
    return *this;
  }
  INLINE_OP nimber_int
  operator+ (const nimber_int &a) const
  {
    return nimber_int (*this) += a;
  }
  INLINE_OP nimber_int
  operator- (const nimber_int &a) const
  {
    return nimber_int (*this) -= a;
  }
#undef INLINE_OP
  friend std::istream &
  operator>> (std::istream &in, nimber_int &n_int)
  {
    in >> n_int.val;
    return in;
  }
  friend std::ostream &
  operator<< (std::ostream &out, const nimber_int &n_int)
  {
    out << n_int.val;
    return out;
  }
  nimber_int
  operator*= (const nimber_int &k)
  {
    if (!val || !k.val)
      {
        val = 0;
        return *this;
      }
    ui a = (val >> 16), b = (val & ((1 << 16) - 1)), c = (k.val >> 16),
       d  = (k.val & ((1 << 16) - 1));
    ui ac = __nimber_mul_4 (a, c), bd = __nimber_mul_4 (b, d);
    ui r1 = __nimber_mul_4 (a ^ b, c ^ d), r2 = __nimber_mul_4 ((1 << 15), ac);
    val = ((r1 ^ bd) << 16) + (r2 ^ bd);
    return *this;
  }
  nimber_int
  operator* (const nimber_int &k) const
  {
    return nimber_int (*this) *= k;
  }
  nimber_int
  inv () const
  {
    nimber_int off = (*this) * (*this), ans = nimber_int (1);
    for (ui i = 0; i < 31; ++i)
      {
        ans *= off, off *= off;
      }
    return ans;
  }
  nimber_int &
  operator/= (const nimber_int &a)
  {
    (*this) *= a.inv ();
    return *this;
  }
  nimber_int
  operator/ (const nimber_int &a) const
  {
    return nimber_int (*this) /= a;
  }
};
typedef nimber_int ni;
}
using game::ni;
}

#endif