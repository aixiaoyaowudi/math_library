/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <power_series_ring/polynomial_kernel_interface.h>
#include <cstring>
#include <iostream>

ntt_kernel::ntt_kernel(std::uint32_t max_conv_size,std::uint32_t P0,std::uint32_t G){
	init(max_conv_size,P0,G);
}

void ntt_kernel::init(std::uint32_t max_conv_size,std::uint32_t P0,std::uint32_t G){
	ker.init(max_conv_size,P0,G);
	P=P0;
	li=math::lmi(P);
}

std::uint32_t ntt_kernel::get_P(){return P;}

std::vector<long long> ntt_kernel::test(std::uint32_t T){
	auto k=ker.test(T);
	return std::vector<long long>(k.begin(),k.end());
}

std::vector<std::uint32_t> ntt_kernel::mul(std::vector<std::uint32_t> ntt_kernel_poly_input1,std::vector<std::uint32_t> ntt_kernel_poly_input2){
	size_t l1,l2,l3;
	std::vector<math::mi> input1(l1=ntt_kernel_poly_input1.size()),input2(l2=ntt_kernel_poly_input2.size());
	for(size_t i=0;i<l1;++i) input1[i]=math::ui2mi(li.v(ntt_kernel_poly_input1[i]));
	for(size_t i=0;i<l2;++i) input2[i]=math::ui2mi(li.v(ntt_kernel_poly_input2[i]));
	std::vector<math::mi> res=ker.mul(input1,input2);
	std::vector<std::uint32_t> ret(l3=res.size());
	for(size_t i=0;i<l3;++i) ret[i]=li.rv(res[i].get_val());
	return ret;
}

ntt_kernel::~ntt_kernel(){
}