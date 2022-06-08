/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <power_series_ring/polynomial_kernel_interface.h>
#include <cstring>
#include <iostream>
#include <exception>

ntt_kernel::ntt_kernel(std::uint32_t max_conv_size,std::uint32_t P0,std::uint32_t G){
	init(max_conv_size,P0,G);
}

ntt_kernel::ntt_kernel(){
	P=0;
}

void ntt_kernel::init(std::uint32_t max_conv_size,std::uint32_t P0,std::uint32_t G){
	try{
		ker.init(max_conv_size,P0,G);
	}catch(std::exception &e){
		P=0;
		throw std::runtime_error(e.what());
	}catch(...){
		P=0;
		throw std::runtime_error("invalid parameters!");
	}
	P=P0;
	li=math::lmi(P);
}

std::uint32_t ntt_kernel::get_P(){return P;}

math::lmi ntt_kernel::get_li(){return li;}

std::vector<long long> ntt_kernel::test(std::uint32_t T){
	auto k=ker.test(T);
	return std::vector<long long>(k.begin(),k.end());
}

std::vector<math::mi> ntt_kernel::mul(std::vector<math::mi> ntt_kernel_poly_input1,std::vector<math::mi> ntt_kernel_poly_input2){
	return ker.mul(ntt_kernel_poly_input1,ntt_kernel_poly_input2);
}

ntt_kernel::~ntt_kernel(){
}