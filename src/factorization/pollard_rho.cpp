/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <factorization/pollard_rho.h>
#include <factorization/miller_rabin.h>
#include <basic/gcd.h>
#include <chrono>
#include <algorithm>

namespace math
{
	namespace factorization
	{
		pollard_rho_random_engine pollard_rho_random_device(std::chrono::high_resolution_clock::now().time_since_epoch().count());
		static ull pollard_rho_u64(ull x)
		{
			ull s=0,t=0,c=((ull)pollard_rho_random_device())%(x-1)+1,j=0,k=1,tmp=1;
			for(k=1;;k<<=1,s=t,tmp=1){
				for(j=1;j<=k;++j){
					t=((u128)t*t+c)%x;
					tmp=(u128)tmp*std::abs((ll)t-(ll)s)%x;
					if((j%127)==0){
						ull d=basic::gcd_u64(tmp,x);
						if(d>1) return d;
					}
				}
				ull d=basic::gcd_u64(tmp,x);
				if(d>1) return d;
			}
			return 1;
		}
		static ui pollard_rho_u32(ui x)
		{
			ui s=0,t=0,c=((ui)pollard_rho_random_device())%(x-1)+1,j=0,k=1,tmp=1;
			for(k=1;;k<<=1,s=t,tmp=1){
				for(j=1;j<=k;++j){
					t=((ull)t*t+c)%x;
					tmp=(ull)tmp*std::abs((i32)t-(i32)s)%x;
					if((j%127)==0){
						ui d=basic::gcd_u32(tmp,x);
						if(d>1) return d;
					}
				}
				ui d=basic::gcd_u32(tmp,x);
				if(d>1) return d;
			}
			return 1;
		}
		static void _factorize_u64(ull n,ui cnt,std::vector<ull> &pms){
			if(n<2) return;
			if(miller_rabin_u64(n)){
				for(ui i=0;i<cnt;++i) pms.push_back(n);
				return;
			}
			ull p=n;
			while(p>=n) p=pollard_rho_u64(n);ui c=0;
			while((n%p)==0) n/=p,++c;
			_factorize_u64(n,cnt,pms),_factorize_u64(p,cnt*c,pms);
		}
		static void _factorize_u32(ui n,ui cnt,std::vector<ui> &pms){
			if(n<2) return;
			if(miller_rabin_u32(n)){
				for(ui i=0;i<cnt;++i) pms.push_back(n);
				return;
			}
			ui p=n;
			while(p>=n) p=pollard_rho_u32(n);ui c=0;
			while((n%p)==0) n/=p,++c;
			_factorize_u32(n,cnt,pms),_factorize_u32(p,cnt*c,pms);
		}
		std::vector<std::pair<ull,ui> > pollard_rho_factorize_u64(ull k){
			std::vector<ull> pms;_factorize_u64(k,1,pms);
			std::sort(pms.begin(),pms.end());std::vector<std::pair<ull,ui>> res;
			for(ui i=0,j;i<pms.size();i=j){
				for(j=i;j<pms.size() && pms[j]==pms[i];++j);
				res.push_back(std::make_pair(pms[i],j-i));
			}
			return res;
		}
		std::vector<std::pair<ui,ui> > pollard_rho_factorize_u32(ui k){
			std::vector<ui> pms;_factorize_u32(k,1,pms);
			std::sort(pms.begin(),pms.end());std::vector<std::pair<ui,ui>> res;
			for(ui i=0,j;i<pms.size();i=j){
				for(j=i;j<pms.size() && pms[j]==pms[i];++j);
				res.push_back(std::make_pair(pms[i],j-i));
			}
			return res;
		}
	}
}