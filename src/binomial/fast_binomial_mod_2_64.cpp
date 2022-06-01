/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <binomial/fast_binomial_mod_2_64.h>
#include <basic/fast_pow.h>
#include <binomial/constants/coefs_for_fast_binomial_2_64>

namespace math
{
	static constexpr __uint128_t coefs[32][32]=coefs_for_fast_binomial_2_64;
	static constexpr ull bases[32]=coefs_for_fast_binomial_2_64_bases;
	static ull calc_bju(ull j,ull u){
		__uint128_t cur=0,u2=__uint128_t(u)*u;
		for(ui i=31;~i;--i) cur=cur*u2+coefs[j][i];cur*=u;
		return basic::fast_pow(bases[j],(ull)(cur>>64));
	}
	static ull odd_factorial(ull k){
		ull ans=1;ull k0=(k>>1);if(k&1) ++k0;
		for(ui j=0;j<32;++j) ans*=calc_bju(j,k0);
		if((ans&2)!=((k0+1)&2)) ans=-ans;
		return ans;
	}
	ull binomial::fast_binomial_mod_2_64::factorial_odd(ull k){
		ull ans=1;
		while(k){ans*=odd_factorial(k);k>>=1;}
		return ans;
	}
	ull binomial::fast_binomial_mod_2_64::factorial(ull k){
		ull cnt2=0;
		while(k){cnt2+=(k>>1);k>>=1;}
		if(cnt2>=64) return 0;
		return factorial_odd(k)<<cnt2;
	}
	ull binomial::fast_binomial_mod_2_64::binomial(ull upper,ull lower){
		if(upper<lower) return 0;ui cnt2=__builtin_popcountll(lower)+__builtin_popcountll(upper-lower)-__builtin_popcountll(upper);
		if(cnt2>=64) return 0;
		ull c=basic::fast_pow(factorial_odd(lower)*factorial_odd(upper-lower),(1ull<<63)-1)*factorial_odd(upper);
		return c<<cnt2;
	}
}