/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_FACTORIZATION_POLLARD_RHO_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_FACTORIZATION_POLLARD_RHO_H_

#include <omp.h>
#include <random>
#include <type/basic_typedef.h>
#include <vector>

namespace math
{
namespace factorization
{
struct pollard_rho_random_engine
{
  std::mt19937_64 random_engine;
  pollard_rho_random_engine (uint seed) : random_engine (seed) {}
  decltype (random_engine ())
  operator() ()
  {
    return random_engine ();
  }
};
extern pollard_rho_random_engine random_engine;
#ifdef _OPENMP
#pragma omp threadprivate(random_engine)
#endif
std::vector<std::pair<ui, ui> > pollard_rho_factorize_u32 (ui k);
std::vector<std::pair<ull, ui> > pollard_rho_factorize_u64 (ull k);
}
}

#endif