/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <factorization/miller_rabin.h>
#include <basic/fast_pow.h>

namespace math
{
	namespace factorization
	{
		static constexpr ui miller_rabin_bases_ui[]={2,7,61};
		static constexpr ull miller_rabin_bases_ull[]={2,325,9375,28178,450775,9780504,1795265022};
		bool miller_rabin_u32(ui k){
			for(ui c:miller_rabin_bases_ui) if(k==c) return true;
			for(ui c:miller_rabin_bases_ui){
				c=c%k;if(c==0) continue;
				ui d=k-1,res=basic::fast_pow(c,d,k);
				if(res!=1) return false;
				while(d&1^1){
					if((d>>=1),((res=basic::fast_pow(c,d,k))==k-1)) break;
					else if(res!=1){
						return false;
					}
				}
			}
			return true;
		}
		bool miller_rabin_u64(ull k){
			for(ull c:miller_rabin_bases_ull) if(k==c) return k==2;
			for(ull c:miller_rabin_bases_ull){
				c=c%k;if(c==0) continue;
				ull d=k-1,res=basic::fast_pow(c,d,k);
				if(res!=1) return false;
				while(d&1^1){
					if((d>>=1),((res=basic::fast_pow(c,d,k))==k-1)) break;
					else if(res!=1){
						return false;
					}
				}
			}
			return true;
		}
		ui miller_rabin_next_prime_u32(ui k){
			while(!miller_rabin_u32(k)) ++k;
			return k;
		}
		ull miller_rabin_next_prime_u64(ull k){
			while(!miller_rabin_u64(k)) ++k;
			return k;
		}
	}
}