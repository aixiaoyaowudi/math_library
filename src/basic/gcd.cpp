/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <algorithm>
#include <basic/gcd.h>
#include <cmath>

namespace math
{
namespace basic
{
ui
gcd_u32 (ui a, ui b)
{
  if (!a || !b)
    return a | b;
  ui t = __builtin_ctz (a | b);
  a >>= __builtin_ctz (a);
  do
    {
      b >>= __builtin_ctz (b);
      if (a > b)
        std::swap (a, b);
      b -= a;
    }
  while (b);
  return a << t;
}
i32
gcd_i32 (i32 a, i32 b)
{
  return gcd_u32 (std::abs (a), std::abs (b));
}
ull
gcd_u64 (ull a, ull b)
{
  if (!a || !b)
    return a | b;
  ui t = __builtin_ctzll (a | b);
  a >>= __builtin_ctzll (a);
  do
    {
      b >>= __builtin_ctzll (b);
      if (a > b)
        std::swap (a, b);
      b -= a;
    }
  while (b);
  return a << t;
}
ll
gcd_i64 (ll a, ll b)
{
  return gcd_u64 (std::abs (a), std::abs (b));
}
}
}