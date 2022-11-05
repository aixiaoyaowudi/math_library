/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <binomial/linear_modulo_preprocessing.h>
#include <cstring>

namespace math
{
binomial::linear_modulo_preprocessing::linear_modulo_preprocessing () {}
binomial::linear_modulo_preprocessing::~linear_modulo_preprocessing ()
{
  release ();
}
void
binomial::linear_modulo_preprocessing::release ()
{
  if (_inv)
    {
      _inv.reset ();
      fac.reset ();
      ifac.reset ();
    }
}
void
binomial::linear_modulo_preprocessing::init (uint maxn, uint P0)
{
  release ();
  rg      = maxn;
  P       = P0;
  fac     = std::make_unique<mi[]> (rg + 32);
  ifac    = std::make_unique<mi[]> (rg + 32);
  _inv    = std::make_unique<mi[]> (rg + 32);
  _inv[0] = 0, _inv[1] = fac[0] = ifac[0] = 1;
  for (uint i = 2; i < rg + 32; ++i)
    _inv[i] = (-mi (P / i)) * _inv[P % i];
  for (uint i = 1; i < rg + 32; ++i)
    fac[i] = fac[i - 1] * mi (i), ifac[i] = ifac[i - 1] * _inv[i];
}
binomial::linear_modulo_preprocessing::linear_modulo_preprocessing (
    const linear_modulo_preprocessing &d)
{
  if (d._inv)
    {
      rg   = d.rg;
      P    = d.P;
      fac  = std::make_unique<mi[]> (rg + 32);
      ifac = std::make_unique<mi[]> (rg + 32);
      _inv = std::make_unique<mi[]> (rg + 32);
      std::memcpy (fac.get (), d.fac.get (), sizeof (mi) * (rg + 32));
      std::memcpy (ifac.get (), d.ifac.get (), sizeof (mi) * (rg + 32));
      memcpy (_inv.get (), d._inv.get (), sizeof (mi) * (rg + 32));
    }
}
}