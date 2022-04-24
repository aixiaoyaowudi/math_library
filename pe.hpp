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
		ull reduce(ull a) {
			ull q = (ull)((L(m) * a) >> 64);
			ull r = a - q * b;
			return r >= b ? r - b : r;
		}
	};
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
		~polynomial();
		void init(uint max_conv_size,uint P0);
		poly mul(const poly &a,const poly &b);
		poly inv(const poly &a);
		poly ln(const poly &a);
		poly exp(const poly &a);
	};
	class fast_binomial_2_64
	{
	private:
		#include <coefs_for_fast_binomial_2_64>
		static constexpr __uint128_t coefs[32][32]=coefs_for_fast_binomial_2_64;
		static constexpr ull bases[32]=coefs_for_fast_binomial_2_64_bases;
		static ull fast_pow(ull a,ull b);
		static ull calc_bju(ull j,ull u);
	public:
		static ull odd_factorial(ull k);
		static ull factorial_odd(ull k);
		static ull factorial(ull k);
		static ull binomial(ull upper,ull lower);
	};
	class fast_binomial_2_32
	{
	private:
		#include <coefs_for_fast_binomial_2_32>
		static constexpr ull coefs[16][16]=coefs_for_fast_binomial_2_32;
		static constexpr uint bases[16]=coefs_for_fast_binomial_2_32_bases;
		static uint fast_pow(uint a,uint b);
		static uint calc_bju(uint j,uint u);
	public:
		static uint odd_factorial(uint k);
		static uint factorial_odd(uint k);
		static uint factorial(uint k);
		static uint binomial(uint upper,uint lower);
	};
	class LinearModuloPreprocessing
	{
	private:
		std::unique_ptr<mint[]> fac,ifac,_inv;uint rg,P;
		void release();
	public:
		LinearModuloPreprocessing();
		LinearModuloPreprocessing(const LinearModuloPreprocessing &d);
		~LinearModuloPreprocessing();
		void init(uint maxn,uint P);
		mint factorial(uint i);
		mint inverse_factorial(uint i);
		mint inverse(uint i);
		mint binomial(uint upper,uint lower);
	};
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