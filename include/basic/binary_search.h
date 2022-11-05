/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_BASIC_BINARY_SEARCH_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_BASIC_BINARY_SEARCH_H_

#include <type/type.h>
#include <type_traits>

namespace math
{
namespace basic
{
template <typename B, typename F, typename... Args>
typename std::enable_if<
    traits::func_call_valid_v<
        F, bool, std::remove_cv_t<std::remove_reference_t<B> >,
        Args...> && std::is_integral_v<std::remove_cv_t<std::remove_reference_t<B> > >,
    std::remove_cv_t<std::remove_reference_t<B> > >::type
binary_search_min (B l0, B r0, F &&func, Args &&... args)
{
  typedef typename std::remove_cv_t<std::remove_reference_t<B> > BB;
  BB ans = r0 + 1, mid, l = l0, r = r0;
  while (l <= r)
    {
      mid = (l + r) / 2;
      if (func (mid, args...))
        ans = mid, r = mid - 1;
      else
        l = mid + 1;
    }
  return ans;
}
template <typename B, typename F, typename... Args>
typename std::enable_if<
    traits::func_call_valid_v<
        F, bool, std::remove_cv_t<std::remove_reference_t<B> >,
        Args...> && std::is_integral_v<std::remove_cv_t<std::remove_reference_t<B> > >,
    std::remove_cv_t<std::remove_reference_t<B> > >::type
binary_search_max (B l0, B r0, F &&func, Args &&... args)
{
  typedef typename std::remove_cv_t<std::remove_reference_t<B> > BB;
  BB ans = l0 - 1, mid, l = l0, r = r0;
  while (l <= r)
    {
      mid = (l + r) / 2;
      if (func (mid, args...))
        ans = mid, l = mid + 1;
      else
        r = mid - 1;
    }
  return ans;
}
template <typename B, typename F, typename... Args>
typename std::enable_if<
    traits::func_call_valid_v<
        F, bool, std::remove_cv_t<std::remove_reference_t<B> >,
        Args...> && std::is_floating_point_v<std::remove_cv_t<std::remove_reference_t<B> > >,
    std::remove_cv_t<std::remove_reference_t<B> > >::type
binary_search_min (B l0, B r0, F &&func, Args &&... args)
{
  typedef typename std::remove_cv_t<std::remove_reference_t<B> > BB;
  BB ans = r0, mid, l = l0, r = r0;
  constexpr double eps = 1e-12;
  while (r - l >= eps)
    {
      mid = (l + r) / 2;
      if (func (mid, args...))
        ans = mid, r = mid;
      else
        l = mid;
    }
  return ans;
}
template <typename B, typename F, typename... Args>
typename std::enable_if<
    traits::func_call_valid_v<
        F, bool, std::remove_cv_t<std::remove_reference_t<B> >,
        Args...> && std::is_floating_point_v<std::remove_cv_t<std::remove_reference_t<B> > >,
    std::remove_cv_t<std::remove_reference_t<B> > >::type
binary_search_max (B l0, B r0, F &&func, Args &&... args)
{
  typedef typename std::remove_cv_t<std::remove_reference_t<B> > BB;
  BB ans = l0, mid, l = l0, r = r0;
  constexpr double eps = 1e-12;
  while (r - l >= eps)
    {
      mid = (l + r) / 2;
      if (func (mid, args...))
        ans = mid, l = mid;
      else
        r = mid;
    }
  return ans;
}
template <typename B, typename F, typename... Args>
typename std::enable_if<
    (!traits::func_call_valid_v<B, bool,
                                std::remove_cv_t<std::remove_reference_t<B> >,
                                F, Args...>)&&traits::
            func_call_valid_v<
                F, bool, std::remove_cv_t<std::remove_reference_t<B> >,
                Args...> && std::is_floating_point_v<std::remove_cv_t<std::remove_reference_t<B> > >,
    std::remove_cv_t<std::remove_reference_t<B> > >::type
binary_search_min (B l0, B r0, B e, F &&func, Args &&... args)
{
  typedef typename std::remove_cv_t<std::remove_reference_t<B> > BB;
  BB ans = r0, mid, l = l0, r = r0;
  while (r - l >= e)
    {
      mid = (l + r) / 2;
      if (func (mid, args...))
        ans = mid, r = mid;
      else
        l = mid;
    }
  return ans;
}
template <typename B, typename F, typename... Args>
typename std::enable_if<
    (!traits::func_call_valid_v<B, bool,
                                std::remove_cv_t<std::remove_reference_t<B> >,
                                F, Args...>)&&traits::
            func_call_valid_v<
                F, bool, std::remove_cv_t<std::remove_reference_t<B> >,
                Args...> && std::is_floating_point_v<std::remove_cv_t<std::remove_reference_t<B> > >,
    std::remove_cv_t<std::remove_reference_t<B> > >::type
binary_search_max (B l0, B r0, B e, F &&func, Args &&... args)
{
  typedef typename std::remove_cv_t<std::remove_reference_t<B> > BB;
  BB ans = l0, mid, l = l0, r = r0;
  constexpr double eps = 1e-12;
  while (r - l > eps)
    {
      mid = (l + r) / 2;
      if (func (mid, args...))
        ans = mid, l = mid;
      else
        r = mid;
    }
  return ans;
}
}
}

#endif