#include <math_library.h>
#include <assert.h>
#include <random>
#include <iostream>
#include <thread>
#include <chrono>
#if defined(_OPENMP)
#include <omp.h>
#endif
using namespace math;
int main(){
	{
		std::cerr<<"Start prime & factorization test"<<std::endl;ui cnt=0;
		{
			std::mt19937_64 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
			constexpr ull rgl=1e14,rgr=2e14,T=20000;
			std::uniform_int_distribution<ull> rnd{rgl,rgr};
			for(ui i=0;i<T;++i){
				ull k=rnd(rng);
				bool mlj=factorization::miller_rabin_u64(k);
				auto fd=factorization::pollard_rho_factorize_u64(k);
				bool cap=true;
				std::vector<std::pair<ull,ui>> md;
				for(ull i=2;i*i<=k;++i) if((k%i)==0){
					ui c=0;
					while((k%i)==0) k/=i,++c;
					md.push_back(std::make_pair(i,c));
					cap=false;
				}
				if(k!=1) md.push_back(std::make_pair(k,1));
				assert(cap==mlj);
				assert(fd==md);
				cnt+=(cap?1:0);
			}
		}
		{
			std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
			constexpr ui rgl=1e9,rgr=2e9,T=20000;
			std::uniform_int_distribution<ui> rnd{rgl,rgr};
			for(ui i=0;i<T;++i){
				ui k=rnd(rng);
				bool mlj=factorization::miller_rabin_u32(k);
				auto fd=factorization::pollard_rho_factorize_u32(k);
				bool cap=true;
				std::vector<std::pair<ui,ui>> md;
				for(ui i=2;i*i<=k;++i) if((k%i)==0){
					ui c=0;
					while((k%i)==0) k/=i,++c;
					md.push_back(std::make_pair(i,c));
					cap=false;
				}
				if(k!=1) md.push_back(std::make_pair(k,1));
				assert(cap==mlj);
				assert(fd==md);
				cnt+=(cap?1:0);
			}
		}
		std::cerr<<"Total:"<<cnt<<" primes.\nEnd prime test"<<std::endl;
	}
	{
		std::cerr<<"Start single thread NTT test"<<std::endl;
		constexpr ui test_size=(1<<20),T=20;
		power_series_ring::polynomial_kernel::polynomial_kernel_ntt p((test_size<<1),default_mod,3);
		constexpr ui other_mod=1000000007;
		modulo::modint::set_mod_mi(other_mod);
		auto r=p.test(T);
		modulo::modint::set_mod_mi(default_mod);
		std::cerr<<"NTT test of size "<<test_size<<std::endl;
		std::cerr<<"Dif                           x"<<T<<" finished in "<<r[0]<<"us"<<std::endl;
		std::cerr<<"Dit                           x"<<T<<" finished in "<<r[1]<<"us"<<std::endl;
		std::cerr<<"Inv                           x"<<T<<" finished in "<<r[2]<<"us"<<std::endl;
		std::cerr<<"Inv(faster)                   x"<<T<<" finished in "<<r[3]<<"us"<<std::endl;
		std::cerr<<"Ln                            x"<<T<<" finished in "<<r[4]<<"us"<<std::endl;
		std::cerr<<"Ln(faster)                    x"<<T<<" finished in "<<r[5]<<"us"<<std::endl;
		std::cerr<<"Exp                           x"<<T<<" finished in "<<r[6]<<"us"<<std::endl;
		std::cerr<<"Multipoint eval interpolation x"<<T<<" finished in "<<r[7]<<"us"<<std::endl;
		std::cerr<<"Lagrange interpolation        x"<<T<<" finished in "<<r[8]<<"us"<<std::endl;
		std::cerr<<"End single thread NTT test"<<std::endl;
	}
	#if defined(_OPENMP)
	{
		std::cerr<<"Start multiple thread NTT test"<<std::endl;
		std::array<long long,9> time_cost={0,0,0,0,0,0,0,0,0};int nt;
		constexpr ui test_size=(1<<20),T=20;
		power_series_ring::polynomial_kernel::polynomial_kernel_ntt p((test_size<<1),default_mod,3);
		constexpr ui other_mod=1000000007;
		modulo::modint::set_mod_for_all_threads_mi(other_mod);
		#pragma omp parallel firstprivate(p)
		{
			auto r=p.test(T);
			#pragma omp critical
			{
				for(ui i=0;i<9;++i) time_cost[i]+=r[i];
				nt=omp_get_num_threads();
			}
		}
		modulo::modint::set_mod_for_all_threads_mi(default_mod);
		std::cerr<<"NTT test of size "<<test_size<<std::endl;
		std::cerr<<"Dif                           x"<<T<<" finished in "<<time_cost[0]/nt<<"us in average"<<std::endl;
		std::cerr<<"Dit                           x"<<T<<" finished in "<<time_cost[1]/nt<<"us in average"<<std::endl;
		std::cerr<<"Inv                           x"<<T<<" finished in "<<time_cost[2]/nt<<"us in average"<<std::endl;
		std::cerr<<"Inv(faster)                   x"<<T<<" finished in "<<time_cost[3]/nt<<"us in average"<<std::endl;
		std::cerr<<"Ln                            x"<<T<<" finished in "<<time_cost[4]/nt<<"us in average"<<std::endl;
		std::cerr<<"Ln(faster)                    x"<<T<<" finished in "<<time_cost[5]/nt<<"us in average"<<std::endl;
		std::cerr<<"Exp                           x"<<T<<" finished in "<<time_cost[6]/nt<<"us in average"<<std::endl;
		std::cerr<<"Multipoint eval interpolation x"<<T<<" finished in "<<time_cost[7]/nt<<"us in average"<<std::endl;
		std::cerr<<"Lagrange interpolation        x"<<T<<" finished in "<<time_cost[8]/nt<<"us in average"<<std::endl;
		std::cerr<<"End multiple thread NTT test"<<std::endl;
	}
	#endif
	std::this_thread::sleep_for(std::chrono::seconds(2));
	{
		std::cerr<<"Start mod_int test"<<std::endl;
		using namespace modulo::modint;
		{
			constexpr ui  omp_test_mod_mi  = 1000000007,
			              omp_calc_mod_mi  = 1000000009;
			constexpr ull omp_test_mod_mli = 1000000000000000177,
			              omp_calc_mod_mli = 1000000000000000183;
			constexpr ui  omp_test_mod_mai = 1000000207,
			              omp_calc_mod_mai = 1000000223;
			constexpr ui  omp_test_mod_m5i = 1000000241,
			              omp_calc_mod_m5i = 1000000271;
			constexpr ui test_num=10000000;
			#if defined(_OPENMP)
			#pragma omp parallel
			#endif
			{
				assert(global_mod_mi==default_mod);
				assert(global_mod_mli==default_mod);
				// #if defined(__AVX__) && defined(__AVX2__)
				// assert(global_mod_mai==default_mod);
				// set_mod_mai(omp_test_mod_mai);
				// assert(global_mod_mai==omp_test_mod_mai);
				// #endif
				// #if defined(__AVX512F__) && defined(__AVX512DQ__)
				// assert(global_mod_m5i==default_mod);
				// set_mod_m5i(omp_test_mod_m5i);
				// assert(global_mod_m5i==omp_test_mod_m5i);
				// #endif
				set_mod_mi(omp_test_mod_mi);
				assert(global_mod_mi==omp_test_mod_mi);
				set_mod_mli(omp_test_mod_mli);
				assert(global_mod_mli==omp_test_mod_mli);
			}
			set_mod_for_all_threads_mi(omp_calc_mod_mi);
			set_mod_for_all_threads_mli(omp_calc_mod_mli);
			// #if defined(__AVX__) && defined(__AVX2__)
			// set_mod_for_all_threads_mai(omp_calc_mod_mai);
			// #endif
			// #if defined(__AVX512F__) && defined(__AVX512DQ__)
			// set_mod_for_all_threads_m5i(omp_calc_mod_m5i);
			// #endif
			#if defined(_OPENMP)
			#pragma omp parallel
			#endif
			{
				std::mt19937	mi_rng (std::chrono::high_resolution_clock::now().time_since_epoch().count())/*,
								mai_rng(std::chrono::high_resolution_clock::now().time_since_epoch().count()),
								m5i_rng(std::chrono::high_resolution_clock::now().time_since_epoch().count())*/;
				std::mt19937_64 mli_rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
				std::uniform_int_distribution<ui> mi_uid{0,omp_calc_mod_mi-1}/*,
												  mai_uid{0,omp_calc_mod_mai-1},
												  m5i_uid{0,omp_calc_mod_m5i-1}*/;
				std::uniform_int_distribution<ull> mli_uid{0,omp_calc_mod_mli-1};
				assert(global_mod_mi==omp_calc_mod_mi);
				assert(global_mod_mli==omp_calc_mod_mli);
				// #if defined(__AVX__) && defined(__AVX2__)
				// assert(global_mod_mai==omp_calc_mod_mai);
				// #endif
				// #if defined(__AVX512F__) && defined(__AVX512DQ__)
				// assert(global_mod_m5i==omp_calc_mod_m5i);
				// #endif
				for(ui i=0;i<test_num;++i){
					{
						ui opa=mi_uid(mi_rng),opb=mi_uid(mi_rng);
						mi mpa=mi(opa),mpb=mi(opb);
						ui v1=((opa+opb)%omp_calc_mod_mi),
						   v2=((opa-opb+omp_calc_mod_mi)%omp_calc_mod_mi),
						   v3=(1ull*opa*opb%omp_calc_mod_mi),
						   v4=(omp_calc_mod_mi-opa)%omp_calc_mod_mi;
						assert(v1==(mpa+mpb).real_val());
						assert(v2==(mpa-mpb).real_val());
						assert(v3==(mpa*mpb).real_val());
						assert(v4==(-mpa).real_val());
						mi mpc=mpa;
						mpc+=mpb;
						assert(v1==mpc.real_val());
						mpc=mpa;
						mpc-=mpb;
						assert(v2==mpc.real_val());
						mpc=mpa;
						mpc*=mpb;
						assert(v3==mpc.real_val());
					}
					{
						ull opa=mli_uid(mli_rng),opb=mli_uid(mli_rng);
						mli mpa=mli(opa),mpb=mli(opb);
						ull v1=((opa+opb)%omp_calc_mod_mli),
						    v2=((opa-opb+omp_calc_mod_mli)%omp_calc_mod_mli),
						    v3=((u128)opa*opb%omp_calc_mod_mli),
						    v4=(omp_calc_mod_mli-opa)%omp_calc_mod_mli;
						assert(v1==(mpa+mpb).real_val());
						assert(v2==(mpa-mpb).real_val());
						assert(v3==(mpa*mpb).real_val());
						assert(v4==(-mpa).real_val());
						mli mpc=mpa;
						mpc+=mpb;
						assert(v1==mpc.real_val());
						mpc=mpa;
						mpc-=mpb;
						assert(v2==mpc.real_val());
						mpc=mpa;
						mpc*=mpb;
						assert(v3==mpc.real_val());
					}
					// #if defined(__AVX__) && defined(__AVX2__)
					// {
					// 	auto c=[](__m256i a,__m256i b)->int{
					// 		__m256i x=_mm256_xor_si256(a,b);
					// 		return _mm256_testz_si256(x,x);
					// 	};
					// 	ui opal[8],opbl[8],v1l[8],v2l[8],v3l[8],v4l[8];
					// 	for(ui j=0;j<8;++j){
					// 		opal[j]=mai_uid(mai_rng);
					// 		opbl[j]=mai_uid(mai_rng);
					// 		v1l[j]=((opal[j]+opbl[j])%omp_calc_mod_mai),
					// 		v2l[j]=((opal[j]-opbl[j]+omp_calc_mod_mai)%omp_calc_mod_mai),
					// 		v3l[j]=(1ull*opal[j]*opbl[j]%omp_calc_mod_mai),
					// 		v4l[j]=(omp_calc_mod_mai-opal[j])%omp_calc_mod_mai;
					// 	}
					// 	__m256i opa=_mm256_loadu_si256((__m256i*)opal),
					// 	        opb=_mm256_loadu_si256((__m256i*)opbl),
					// 	        v1 =_mm256_loadu_si256((__m256i*)v1l),
					// 	        v2 =_mm256_loadu_si256((__m256i*)v2l),
					// 	        v3 =_mm256_loadu_si256((__m256i*)v3l),
					// 	        v4 =_mm256_loadu_si256((__m256i*)v4l);
					// 	mai mpa(opa),mpb(opb);
					// 	assert(c(v1,(mpa+mpb).real_val()));
					// 	assert(c(v2,(mpa-mpb).real_val()));
					// 	assert(c(v3,(mpa*mpb).real_val()));
					// 	assert(c(v4,(-mpa).real_val()));
					// 	mai mpc=mpa;
					// 	mpc+=mpb;
					// 	assert(c(v1,mpc.real_val()));
					// 	mpc=mpa;
					// 	mpc-=mpb;
					// 	assert(c(v2,mpc.real_val()));
					// 	mpc=mpa;
					// 	mpc*=mpb;
					// 	assert(c(v3,mpc.real_val()));
					// }
					// #endif
					// #if defined(__AVX512F__) && defined(__AVX512DQ__)
					// {
					// 	auto c=[](__m512i a,__m512i b)->int{
					// 		__m512i x=_mm512_xor_si512(a,b);
					// 		return (_mm512_test_epi32_mask(x,x)==0)?1:0;
					// 	};
					// 	ui opal[16],opbl[16],v1l[16],v2l[16],v3l[16],v4l[16];
					// 	for(ui j=0;j<8;++j){
					// 		opal[j]=m5i_uid(mai_rng);
					// 		opbl[j]=m5i_uid(mai_rng);
					// 		v1l[j]=((opal[j]+opbl[j])%omp_calc_mod_m5i),
					// 		v2l[j]=((opal[j]-opbl[j]+omp_calc_mod_m5i)%omp_calc_mod_m5i),
					// 		v3l[j]=(1ull*opal[j]*opbl[j]%omp_calc_mod_m5i),
					// 		v4l[j]=(omp_calc_mod_m5i-opal[j])%omp_calc_mod_m5i;
					// 	}
					// 	__m512i opa=_mm512_loadu_si512((__m512i*)opal),
					// 	        opb=_mm512_loadu_si512((__m512i*)opbl),
					// 	        v1 =_mm512_loadu_si512((__m512i*)v1l),
					// 	        v2 =_mm512_loadu_si512((__m512i*)v2l),
					// 	        v3 =_mm512_loadu_si512((__m512i*)v3l),
					// 	        v4 =_mm512_loadu_si512((__m512i*)v4l);
					// 	m5i mpa(opa),mpb(opb);
					// 	assert(c(v1,(mpa+mpb).real_val()));
					// 	assert(c(v2,(mpa-mpb).real_val()));
					// 	assert(c(v3,(mpa*mpb).real_val()));
					// 	assert(c(v4,(-mpa).real_val()));
					// 	m5i mpc=mpa;
					// 	mpc+=mpb;
					// 	assert(c(v1,mpc.real_val()));
					// 	mpc=mpa;
					// 	mpc-=mpb;
					// 	assert(c(v2,mpc.real_val()));
					// 	mpc=mpa;
					// 	mpc*=mpb;
					// 	assert(c(v3,mpc.real_val()));
					// }
					// #endif
				}
			}
			#if defined(_OPENMP)
			std::cerr<<"mod_int test under openmp finished."<<std::endl;
			#else
			std::cerr<<"mod_int test finished."<<std::endl;
			#endif
		}
		std::cerr<<"End mod_int test"<<std::endl;
	}
}