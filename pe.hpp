/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

/* macros:
* DO_NOT_IGNORE_WARNINGS:
*     enables -Wunused-result
* NO_PROGRESS:
*     disable progress bar
*/

#ifndef _XIAOYAOWUDI_PROJECT_EULER_MATH_LIBRARY_
#define _XIAOYAOWUDI_PROJECT_EULER_MATH_LIBRARY_

#if __cplusplus < 201703L
#error "Require C++17 to Compile"
#endif

#ifndef DO_NOT_IGNORE_WARNINGS
#pragma GCC diagnostic ignored "-Wunused-result"
#endif
#include <immintrin.h>
#include <stdint.h>
#include <iostream>
#include <new>
#ifdef __INTEL_COMPILER
#include <aligned_new>
#endif
#include <vector>
#include <chrono>
#include <thread>
#include <iomanip>
#include <stdexcept>
#include <cstring>
#include <random>
#include <algorithm>
#include <functional>

namespace math
{
	typedef uint32_t uint;
	typedef int64_t ll;
	typedef uint64_t ull;
	typedef __uint128_t L;
	struct FastMod {
		ull b, m;
		FastMod(ull b) : b(b), m(ull((L(1) << 64) / b)) {}
		FastMod(){}
		FastMod(FastMod &d):b(d.b),m(d.m){}
		ull reduce(ull a) {
			ull q = (ull)((L(m) * a) >> 64);
			ull r = a - q * b;
			return r >= b ? r - b : r;
		}
	};
	extern FastMod global_fast_mod;
	#ifdef _OPENMP
	#pragma omp threadprivate(global_fast_mod)
	#endif
	struct montgomery_int_lib{
		constexpr static uint calc_k(uint MOD,uint len){uint ans=1;for(int i=1;i<len;++i) ans=(ans*(MOD+1)+1);return ans;}
		uint P,P2,NP,Pk;static const uint32_t uint_len = sizeof(uint)*8;
		montgomery_int_lib(uint P0):P(P0),P2(P0*2),NP((-ull(P0))%P0),Pk(calc_k(P0,sizeof(uint)*8)){}
		montgomery_int_lib(){}
		uint redd(uint k) const {return k>=P2?k-P2:k;}uint reds(uint k) const {return k>=P?k-P:k;}uint redu(ull k) const {return (k+ull(uint(k)*Pk)*P)>>uint_len;}
	};
	struct montgomery_int{
		static montgomery_int_lib mlib;
		#if defined(_OPENMP)
		#pragma omp threadprivate(mlib)
		#endif
		uint val;
		void init(uint a){val=mlib.redu(ull(a)*mlib.NP);}
		montgomery_int(){val=0;} montgomery_int(const montgomery_int &a):val(a.val){} montgomery_int(uint v):val(mlib.redu(ull(v)*mlib.NP)){}
		montgomery_int& operator=(const montgomery_int &b) {val=b.val;return *this;}
		montgomery_int& operator+=(const montgomery_int &b) {val=mlib.redd(val+b.val);return *this;}
		montgomery_int& operator-=(const montgomery_int &b) {val=mlib.redd(val-b.val+mlib.P2);return *this;}
		montgomery_int operator+(const montgomery_int &b) const {return montgomery_int(*this)+=b;}
		montgomery_int operator-(const montgomery_int &b) const {return montgomery_int(*this)-=b;}
		montgomery_int operator*=(const montgomery_int &b) {val=mlib.redu(ull(val)*b.val);return *this;}
		montgomery_int operator*(const montgomery_int &b) const {return montgomery_int(*this)*=b;}
		montgomery_int operator-() const {montgomery_int b;b.val=mlib.redd(mlib.P2-val);return b;}
		uint real_val() const {return mlib.reds(mlib.redu(val));}
		friend std::istream& operator>>(std::istream &in, montgomery_int &m_int){uint inp;in>>inp;m_int.init(inp);return in;}
		friend std::ostream& operator<<(std::ostream &out, const montgomery_int &m_int){out<<m_int.real_val();return out;}
	};
	constexpr int default_mod=998244353;
	typedef montgomery_int mint;
	extern int global_mod;
	#ifdef _OPENMP
	#pragma omp threadprivate(global_mod)
	#endif
	void set_mint_mod(uint32_t p);
	mint fast_pow(mint a,ull b);
	#if defined(__AVX__) && defined(__AVX2__)
	struct montgomery_mm256_lib{
		alignas(32) __m256i P,P2,NP,Pk,mask1;
		montgomery_mm256_lib(uint P0){P=_mm256_set1_epi32(P0),P2=_mm256_set1_epi32(P0*2),mask1=_mm256_setr_epi32(0,P0*2,0,P0*2,0,P0*2,0,P0*2),
			NP=_mm256_set1_epi32(uint((-ull(P0))%P0)),Pk=_mm256_set1_epi32(montgomery_int_lib::calc_k(P0,sizeof(uint)*8));}
		montgomery_mm256_lib(){}
		#define INLINE_OP __attribute__((__always_inline__))
		INLINE_OP __m256i redd(__m256i k){__m256i a=_mm256_sub_epi32(k,P2);__m256i b=_mm256_and_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(),a),P2);return _mm256_add_epi32(a,b);}
		INLINE_OP __m256i reds(__m256i k){__m256i a=_mm256_sub_epi32(k,P); __m256i b=_mm256_and_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(),a),P); return _mm256_add_epi32(a,b);}
		INLINE_OP __m256i redu(__m256i k){return _mm256_srli_si256(_mm256_add_epi64(_mm256_mul_epu32(_mm256_mul_epu32(k,Pk),P),k),4);}
		INLINE_OP __m256i mul(__m256i k1,__m256i k2){
			return _mm256_or_si256(redu(_mm256_mul_epu32(k1,k2)),_mm256_slli_si256(redu(_mm256_mul_epu32(_mm256_srli_si256(k1,4),_mm256_srli_si256(k2,4))),4));}
		INLINE_OP __m256i add(__m256i k1,__m256i k2){return redd(_mm256_add_epi32(k1,k2));}
		INLINE_OP __m256i sub(__m256i k1,__m256i k2){return redd(_mm256_add_epi32(P2,_mm256_sub_epi32(k1,k2)));}
		#undef INLINE_OP
	};
	struct montgomery_mm256_int{
		static montgomery_mm256_lib mlib;
		#if defined(_OPENMP)
		#pragma omp threadprivate(mlib)
		#endif
		__m256i val;
		void init(__m256i a){val=mlib.mul(a,mlib.NP);}
		montgomery_mm256_int(){val=_mm256_set1_epi32(0);} montgomery_mm256_int(const montgomery_mm256_int &a):val(a.val){} montgomery_mm256_int(__m256i v):val(mlib.mul(v,mlib.NP)){}
		montgomery_mm256_int& operator=(const montgomery_mm256_int &b) {val=b.val;return *this;}
		montgomery_mm256_int& operator+=(const montgomery_mm256_int &b) {val=mlib.add(val,b.val);return *this;}
		montgomery_mm256_int& operator-=(const montgomery_mm256_int &b) {val=mlib.sub(val,b.val);return *this;}
		montgomery_mm256_int operator+(const montgomery_mm256_int &b) const {return montgomery_mm256_int(*this)+=b;}
		montgomery_mm256_int operator-(const montgomery_mm256_int &b) const {return montgomery_mm256_int(*this)-=b;}
		montgomery_mm256_int operator*=(const montgomery_mm256_int &b) {val=mlib.mul(val,b.val);return *this;}
		montgomery_mm256_int operator*(const montgomery_mm256_int &b) const {return montgomery_mm256_int(*this)*=b;}
		__m256i real_val() const {return mlib.reds(_mm256_or_si256(mlib.redu(_mm256_blend_epi32(_mm256_srli_si256(val,4),_mm256_setzero_si256(),0xAA)),
			_mm256_slli_si256(mlib.redu(_mm256_blend_epi32(val,_mm256_setzero_si256(),0xAA)),4)));}
	};
	typedef montgomery_mm256_int m256int;
	void set_m256int_mod(uint32_t p);
	#endif
	void set_mod_for_all_threads(uint32_t p);
	void set_mod(uint32_t p);
	typedef std::vector<mint> poly;
	template<typename T,size_t align_val>
	struct aligned_delete {
		void operator()(T* ptr) const {
			operator delete[](ptr,std::align_val_t(align_val));
		}
	};
	template<typename T,size_t align_val>
	using aligned_array=std::unique_ptr<T[],aligned_delete<T,align_val>>;
	template<typename T, size_t align_val>
	aligned_array<T,align_val> create_aligned_array(size_t size){
		return aligned_array<T,align_val>(new(std::align_val_t(align_val)) T[size]);
	}
	class polynomial;
	class polynomial_ntt
	{
	private:
		aligned_array<mint,32> ws0,ws1,fac,ifac,_inv,tt;
		aligned_array<uint,32> rev,lgg;uint P,G;
		uint fn,fb;
		void release();
	public:
		friend class polynomial;
		polynomial_ntt(uint max_conv_size,uint P0,uint G0);
		void init(uint max_conv_size,uint P0,uint G0);
		polynomial_ntt(const polynomial_ntt &d);
		polynomial_ntt();
		void NTT(poly &p,int V);
		poly mul(const poly &a,const poly &b);
		poly inv(const poly &a);
		poly ln(const poly &a);
		poly exp(const poly &a);
		~polynomial_ntt();
	};
	class polynomial
	{
	private:
		static constexpr uint P1=469762049,P2=998244353,P3=1004535809,I1=554580198,I2=395249030;
		polynomial_ntt pn1;
		polynomial_ntt pn2;
		polynomial_ntt pn3;
		FastMod F,F1,F2,F3;uint N3,P;
		aligned_array<mint,32> _inv;
	public:
		void release();
		polynomial(const polynomial &d);
		polynomial();
		polynomial(uint max_conv_size,uint P0);
		~polynomial();
		void init(uint max_conv_size,uint P0);
		poly mul(const poly &a,const poly &b);
		poly inv(const poly &a);
		poly ln(const poly &a);
		poly exp(const poly &a);
	};
	namespace fast_binomial_2_64
	{
		#include <coefs_for_fast_binomial_2_64>
		constexpr __uint128_t coefs[32][32]=coefs_for_fast_binomial_2_64;
		constexpr ull bases[32]=coefs_for_fast_binomial_2_64_bases;
		ull fast_pow(ull a,ull b);
		ull calc_bju(ull j,ull u);
		ull odd_factorial(ull k);
		ull factorial_odd(ull k);
		ull factorial(ull k);
		ull binomial(ull upper,ull lower);
	};
	namespace fast_binomial_2_32
	{
		#include <coefs_for_fast_binomial_2_32>
		constexpr ull coefs[16][16]=coefs_for_fast_binomial_2_32;
		constexpr uint bases[16]=coefs_for_fast_binomial_2_32_bases;
		uint fast_pow(uint a,uint b);
		uint calc_bju(uint j,uint u);
		uint odd_factorial(uint k);
		uint factorial_odd(uint k);
		uint factorial(uint k);
		uint binomial(uint upper,uint lower);
	}
	class linear_modulo_preprocessing
	{
	private:
		std::unique_ptr<mint[]> fac,ifac,_inv;uint rg,P;
		void release();
	public:
		linear_modulo_preprocessing();
		linear_modulo_preprocessing(uint maxn,uint P);
		linear_modulo_preprocessing(const linear_modulo_preprocessing &d);
		~linear_modulo_preprocessing();
		void init(uint maxn,uint P);
		mint factorial(uint i) const;
		mint inverse_factorial(uint i) const;
		mint inverse(uint i) const;
		mint binomial(uint upper,uint lower) const;
	};
	constexpr int sieve_prefix_sum_offset=16;
	enum sieve_flag
	{
		sieve_mu=1<<0,
		sieve_euler_phi=1<<1,
		sieve_divisors=1<<2,
		sieve_divisors_sum=(1<<3),
		sieve_prefix_sum_mu=1<<(sieve_prefix_sum_offset+0),
		sieve_prefix_sum_euler_phi=1<<(sieve_prefix_sum_offset+1),
		sieve_prefix_sum_divisors=1<<(sieve_prefix_sum_offset+2),
		sieve_prefix_sum_divisors_sum=1<<(sieve_prefix_sum_offset+3),
		sieve_all=sieve_mu|sieve_euler_phi|sieve_divisors|sieve_divisors_sum|sieve_prefix_sum_mu|sieve_prefix_sum_euler_phi|sieve_prefix_sum_divisors|sieve_prefix_sum_divisors_sum
	};
	class sieve
	{
	private:
		std::unique_ptr<uint[]> mnf,phi,mnfc,pksum,d,ds;uint rg,pc;
		std::unique_ptr<int[]> _mu;
		std::unique_ptr<ll[]> premu,preei,pred,preds;
		std::unique_ptr<bool[]> vis;
		std::vector<uint> ps;
		void release();
	public:
		sieve();
		~sieve();
		sieve(uint maxn,int flag=0);
		sieve(const sieve &t);
		void init(uint maxn,int flag=0);
		uint prime_count() const;
		std::vector<uint> all_primes() const;
		uint nth_prime(uint k) const;
		bool is_prime(uint k) const;
		uint min_prime_factor(uint k) const;
		std::vector<std::pair<uint,uint>> factor(uint k) const;
		int mu(uint k) const;
		uint euler_phi(uint k) const;
		uint divisors(uint k) const;
		uint divisors_sum(uint k) const;
		ll prefix_sum_mu(uint k) const;
		ll prefix_sum_euler_phi(uint k) const;
		ll prefix_sum_divisors(uint k) const;
		ll prefix_sum_divisors_sum(uint k) const;
		uint prime_index(uint k) const;
	};
	extern sieve global_sieve;
	extern uint global_sieve_range;
	extern int global_sieve_flag;
	void set_global_sieve(uint rg,int flag=sieve_all);
	struct _random_engine
	{
		std::mt19937_64 random_engine;
		_random_engine(uint seed):random_engine(seed){}
		decltype(random_engine()) operator()(){return random_engine();}
	};
	extern _random_engine random_engine;
	#ifdef _OPENMP
	#pragma omp threadprivate(random_engine)
	#endif
	namespace basic
	{
		ull gcdll(ull a,ull b);
		uint gcd(uint a,uint b);
	}
	namespace factorization
	{
		ull fast_pow_mod(ull a,ull b,ull c);
		ull fast_pow_without_mod(ull a,uint b);
		constexpr ull bases[]={2,325,9375,28178,450775,9780504,1795265022};
		ull pollard_rho(ull x);
		void _factorize(ull n,uint cnt,std::vector<ull> &pms);
		bool is_prime(ull k);
		std::vector<std::pair<ull,uint> > factor(ull k);
		ull euler_phi(ull k);
		ull euler_phi(const std::vector<std::pair<ull,uint>> &decomp);
		int moebius(ull k);
		int moebius(const std::vector<std::pair<ull,uint>> &decomp);
		std::vector<ull> divisors_set(ull k);
		std::vector<ull> divisors_set(const std::vector<std::pair<ull,uint>> &decomp);
		ull sigma(ull k,uint s);
		ull sigma(const std::vector<std::pair<ull,uint>> &decomp,uint s);
	}
}
namespace tools
{
	class timer{
	private:
		typedef std::chrono::high_resolution_clock clock;
		typedef std::chrono::duration<double, std::ratio<1> > second;
		std::chrono::time_point<clock> start_time;
		double accumulated_time;
		bool running;
	public:
		timer();
		void start();
		double stop();
		double accumulated();
		double lap();
		void reset();
		bool get_state();
	};
	class progress_bar{
	private:
		uint32_t total_work;
		uint32_t next_update;
		uint32_t call_diff;
		uint32_t work_done;
		uint16_t old_percent;
		timer    _timer;
		void clear_console_line() const;
	public:
		void start(uint32_t total_work);
		void update(uint32_t work_done0,bool is_dynamic=true);
		progress_bar& operator++();
		double stop();
		double time_it_took();
		uint32_t cells_processed() const;
		~progress_bar();
	};
}
#endif