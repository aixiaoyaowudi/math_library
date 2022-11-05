/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <cstring>
#include <exception>
#include <iostream>
#include <power_series_ring/polynomial_kernel_interface.h>

ntt_kernel::ntt_kernel (std::uint32_t max_conv_size, std::uint32_t P0,
                        std::uint32_t G)
{
  init (max_conv_size, P0, G);
}

mtt_kernel::mtt_kernel (std::uint32_t max_conv_size, std::uint32_t P0)
{
  init (max_conv_size, P0);
}

ntt_kernel::ntt_kernel () { P = 0; }

mtt_kernel::mtt_kernel () { P = 0; }

void
ntt_kernel::init (std::uint32_t max_conv_size, std::uint32_t P0,
                  std::uint32_t G)
{
  try
    {
      ker.init (max_conv_size, P0, G);
    }
  catch (std::exception &e)
    {
      P = 0;
      throw std::runtime_error (e.what ());
    }
  catch (...)
    {
      P = 0;
      throw std::runtime_error ("invalid parameters!");
    }
  P  = P0;
  li = math::lmi (P);
}

void
mtt_kernel::init (std::uint32_t max_conv_size, std::uint32_t P0)
{
  try
    {
      ker.init (max_conv_size, P0);
    }
  catch (std::exception &e)
    {
      P = 0;
      throw std::runtime_error (e.what ());
    }
  catch (...)
    {
      P = 0;
      throw std::runtime_error ("invalid parameters!");
    }
  P  = P0;
  li = math::lmi (P);
}

std::uint32_t
ntt_kernel::get_P ()
{
  return P;
}

std::uint32_t
mtt_kernel::get_P ()
{
  return P;
}

math::lmi
ntt_kernel::get_li ()
{
  return li;
}

math::lmi
mtt_kernel::get_li ()
{
  return li;
}

#define CHECK_INITED                                                          \
  {                                                                           \
    if (!P)                                                                   \
      throw std::runtime_error ("kernel not inited.");                        \
  }

std::vector<math::mi>
ntt_kernel::mul (std::vector<math::mi> ntt_kernel_poly_input1,
                 std::vector<math::mi> ntt_kernel_poly_input2)
{
  CHECK_INITED;
  return ker.mul (ntt_kernel_poly_input1, ntt_kernel_poly_input2);
}

std::vector<math::mi>
mtt_kernel::mul (std::vector<math::mi> mtt_kernel_poly_input1,
                 std::vector<math::mi> mtt_kernel_poly_input2)
{
  CHECK_INITED;
  return ker.mul (mtt_kernel_poly_input1, mtt_kernel_poly_input2);
}

std::vector<math::mi>
ntt_kernel::add (std::vector<math::mi> ntt_kernel_poly_input1,
                 std::vector<math::mi> ntt_kernel_poly_input2)
{
  CHECK_INITED;
  return ker.add (ntt_kernel_poly_input1, ntt_kernel_poly_input2);
}

std::vector<math::mi>
mtt_kernel::add (std::vector<math::mi> mtt_kernel_poly_input1,
                 std::vector<math::mi> mtt_kernel_poly_input2)
{
  CHECK_INITED;
  return ker.add (mtt_kernel_poly_input1, mtt_kernel_poly_input2);
}

std::vector<math::mi>
ntt_kernel::sub (std::vector<math::mi> ntt_kernel_poly_input1,
                 std::vector<math::mi> ntt_kernel_poly_input2)
{
  CHECK_INITED;
  return ker.sub (ntt_kernel_poly_input1, ntt_kernel_poly_input2);
}

std::vector<math::mi>
mtt_kernel::sub (std::vector<math::mi> mtt_kernel_poly_input1,
                 std::vector<math::mi> mtt_kernel_poly_input2)
{
  CHECK_INITED;
  return ker.sub (mtt_kernel_poly_input1, mtt_kernel_poly_input2);
}

std::vector<math::mi>
ntt_kernel::inv (std::vector<math::mi> ntt_kernel_poly_input1)
{
  CHECK_INITED;
  return ker.inv (ntt_kernel_poly_input1);
}

std::vector<math::mi>
ntt_kernel::ln (std::vector<math::mi> ntt_kernel_poly_input1)
{
  CHECK_INITED;
  return ker.ln (ntt_kernel_poly_input1);
}

std::vector<math::mi>
ntt_kernel::exp (std::vector<math::mi> ntt_kernel_poly_input1)
{
  CHECK_INITED;
  return ker.exp (ntt_kernel_poly_input1);
}

std::vector<math::mi>
ntt_kernel::integrate (std::vector<math::mi> ntt_kernel_poly_input1)
{
  CHECK_INITED;
  return ker.integrate (ntt_kernel_poly_input1);
}

std::vector<math::mi>
ntt_kernel::derivative (std::vector<math::mi> ntt_kernel_poly_input1)
{
  CHECK_INITED;
  return ker.derivative (ntt_kernel_poly_input1);
}

std::vector<math::mi>
ntt_kernel::multipoint_eval_interpolation (
    std::vector<math::mi> ntt_kernel_poly_input1,
    std::vector<math::mi> ntt_kernel_poly_input2)
{
  CHECK_INITED;
  return ker.multipoint_eval_interpolation (ntt_kernel_poly_input1,
                                            ntt_kernel_poly_input2);
}

std::vector<math::mi>
ntt_kernel::lagrange_interpolation (
    std::vector<math::mi> ntt_kernel_poly_input1,
    std::vector<math::mi> ntt_kernel_poly_input2)
{
  CHECK_INITED;
  if (ntt_kernel_poly_input1.size () != ntt_kernel_poly_input2.size ())
    throw std::runtime_error ("both inputs must have the same size!");
  size_t l = ntt_kernel_poly_input1.size ();
  std::vector<std::pair<math::mi, math::mi> > src (l);
  for (size_t i = 0; i < l; ++i)
    src[i] = std::make_pair (ntt_kernel_poly_input1[i],
                             ntt_kernel_poly_input2[i]);
  return ker.lagrange_interpolation (src);
}

#undef CHECK_INITED

ntt_kernel::~ntt_kernel () {}

mtt_kernel::~mtt_kernel () {}