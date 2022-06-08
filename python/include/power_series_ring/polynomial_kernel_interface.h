/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_PYTHON_POLYNOMIAL_KERNEL_INTERFACE_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_PYTHON_POLYNOMIAL_KERNEL_INTERFACE_H_

#include <power_series_ring/polynomial_kernel.h>
#include <modulo/modint.h>
#include <stdint.h>
#include <vector>

class ntt_kernel
{
private:
	math::power_series_ring::polynomial_kernel::polynomial_kernel_ntt ker;
	std::uint32_t P;
	math::lmi li;
public:
	ntt_kernel()=default;
	ntt_kernel(std::uint32_t max_conv_size,std::uint32_t P0,std::uint32_t G);
	void init(std::uint32_t max_conv_size,std::uint32_t P0,std::uint32_t G);
	std::vector<long long> test(std::uint32_t T);
	std::uint32_t get_P();
	std::vector<std::uint32_t> mul(std::vector<std::uint32_t> ntt_kernel_poly_input1,std::vector<std::uint32_t> ntt_kernel_poly_input2);
	~ntt_kernel();
};

#endif