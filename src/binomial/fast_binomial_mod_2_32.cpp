/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <binomial/fast_binomial_mod_2_32.h>
#include <basic/fast_pow.h>

namespace math
{
	ui binomial::fast_binomial_mod_2_32::calc_bju(ui j,ui u){
		ull cur=0,u2=1ull*u*u;
		for(uint i=15;~i;--i) cur=cur*u2+coefs[j][i];cur*=u;
		return basic::fast_pow(bases[j],(ui)(cur>>32));
	}
	ui binomial::fast_binomial_mod_2_32::odd_factorial(ui k){
		ui ans=1;ui k0=(k>>1);if(k&1) ++k0;
		for(int j=0;j<16;++j) ans*=calc_bju(j,k0);
		if((ans&2)!=((k0+1)&2)) ans=-ans;
		return ans;
	}
	ui binomial::fast_binomial_mod_2_32::factorial_odd(ui k){
		ui ans=1;
		while(k){ans*=odd_factorial(k);k>>=1;}
		return ans;
	}
	ui binomial::fast_binomial_mod_2_32::factorial(ui k){
		ui cnt2=0;
		while(k){cnt2+=(k>>1);k>>=1;}
		if(cnt2>=32) return 0;
		return factorial_odd(k)<<cnt2;
	}
	ui binomial::fast_binomial_mod_2_32::binomial(ui upper,ui lower){
		if(upper<lower) return 0;ui cnt2=__builtin_popcount(lower)+__builtin_popcount(upper-lower)-__builtin_popcount(upper);
		if(cnt2>=32) return 0;
		ui c=basic::fast_pow(factorial_odd(lower)*factorial_odd(upper-lower),(1u<<31)-1)*factorial_odd(upper);
		return c<<cnt2;
	}
}