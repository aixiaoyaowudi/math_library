/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_PYTHON_POWER_SERIES_RING_POLYNOMIAL_KERNEL_INTERFACE_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_PYTHON_POWER_SERIES_RING_POLYNOMIAL_KERNEL_INTERFACE_H_

#include <power_series_ring/polynomial_kernel.h>
#include <modulo/modint.h>
#include <stdint.h>
#include <vector>

#ifndef SWIG_NOT_EXPORT
#define SWIG_NOT_EXPORT(x) x
#endif

class ntt_kernel
{
private:
	math::power_series_ring::polynomial_kernel::polynomial_kernel_ntt ker;
	std::uint32_t P;
	math::lmi li;
public:
	ntt_kernel();
	ntt_kernel(std::uint32_t max_conv_size,std::uint32_t P0,std::uint32_t G);
	void init(std::uint32_t max_conv_size,std::uint32_t P0,std::uint32_t G);
	std::uint32_t get_P();
	SWIG_NOT_EXPORT(math::lmi get_li(););
	std::vector<math::mi> mul(std::vector<math::mi> ntt_kernel_poly_input1,std::vector<math::mi> ntt_kernel_poly_input2);
	std::vector<math::mi> add(std::vector<math::mi> ntt_kernel_poly_input1,std::vector<math::mi> ntt_kernel_poly_input2);
	std::vector<math::mi> sub(std::vector<math::mi> ntt_kernel_poly_input1,std::vector<math::mi> ntt_kernel_poly_input2);
	std::vector<math::mi> inv(std::vector<math::mi> ntt_kernel_poly_input1);
	std::vector<math::mi> ln(std::vector<math::mi> ntt_kernel_poly_input1);
	std::vector<math::mi> exp(std::vector<math::mi> ntt_kernel_poly_input1);
	std::vector<math::mi> integrate(std::vector<math::mi> ntt_kernel_poly_input1);
	std::vector<math::mi> derivative(std::vector<math::mi> ntt_kernel_poly_input1);
	std::vector<math::mi> multipoint_eval_interpolation(std::vector<math::mi> ntt_kernel_poly_input1,std::vector<math::mi> ntt_kernel_poly_input2);
	std::vector<math::mi> lagrange_interpolation(std::vector<math::mi> ntt_kernel_poly_input1,std::vector<math::mi> ntt_kernel_poly_input2);
	~ntt_kernel();
};

class mtt_kernel
{
private:
	math::power_series_ring::polynomial_kernel::polynomial_kernel_mtt ker;
	std::uint32_t P;
	math::lmi li;
public:
	mtt_kernel();
	mtt_kernel(std::uint32_t max_conv_size,std::uint32_t P0);
	void init(std::uint32_t max_conv_size,std::uint32_t P0);
	std::uint32_t get_P();
	SWIG_NOT_EXPORT(math::lmi get_li(););
	std::vector<math::mi> mul(std::vector<math::mi> ntt_kernel_poly_input1,std::vector<math::mi> ntt_kernel_poly_input2);
	std::vector<math::mi> add(std::vector<math::mi> ntt_kernel_poly_input1,std::vector<math::mi> ntt_kernel_poly_input2);
	std::vector<math::mi> sub(std::vector<math::mi> ntt_kernel_poly_input1,std::vector<math::mi> ntt_kernel_poly_input2);
	~mtt_kernel();
};

#endif