/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <modulo/modint.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace math
{
namespace modulo
{
namespace modint
{
u32 global_mod_mi = default_mod;
barrett_reduction_u32 global_fast_mod (default_mod);
montgomery_modint_lib mi::mlib (default_mod);
void
set_mod_mi (ui p)
{
  mi::mlib      = montgomery_modint_lib (p);
  global_mod_mi = p;
}
void
set_mod_for_all_threads_mi (ui p)
{
#if defined(_OPENMP)
#pragma omp parallel
  {
#endif
    mi::mlib      = montgomery_modint_lib (p);
    global_mod_mi = p;
#if defined(_OPENMP)
  }
#endif
}
ull global_mod_mli = default_mod;
montgomery_modlongint_lib mli::mlib (default_mod);
void
set_mod_mli (ull p)
{
  mli::mlib      = montgomery_modlongint_lib (p);
  global_mod_mli = p;
}
void
set_mod_for_all_threads_mli (ull p)
{
#if defined(_OPENMP)
#pragma omp parallel
  {
#endif
    mli::mlib      = montgomery_modlongint_lib (p);
    global_mod_mli = p;
#if defined(_OPENMP)
  }
#endif
}
}
}
}