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
#if defined(__INTEL_COMPILER)
#include <aligned_new>
#define assume_aligned(a,b) __assume_aligned((a),(b))
#elif defined(__GNUC__)
#define assume_aligned(a,b) ((a)=__builtin_assume_aligned((a),(b)))
#define restrict
#include <new>
#else
#error "only gnu compiler or intel compiler is supported"
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
#include <ext/pb_ds/assoc_container.hpp>
#include <ext/pb_ds/hash_policy.hpp>
#include <array>
#include <type_traits>
#include <tuple>

namespace math
{
	typedef uint32_t ui;
	typedef int32_t i32;
	typedef int64_t ll;
	typedef uint64_t ull;
	typedef __uint128_t u128;
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
	namespace modulo
	{
		namespace mod_int
		{
			struct fast_mod_32 {
				ull b, m;
				fast_mod_32(ull b) : b(b), m(ull((u128(1) << 64) / b)) {}
				fast_mod_32(){}
				fast_mod_32(fast_mod_32 &d):b(d.b),m(d.m){}
				ull reduce(ull a) {
					ull q = (ull)((u128(m) * a) >> 64);
					ull r = a - q * b;
					return r >= b ? r - b : r;
				}
			};
			extern fast_mod_32 global_fast_mod_32;
			constexpr ui default_mod=998244353;
			struct montgomery_mi_lib{
				constexpr static ui calc_k(ui MOD,ui len){ui ans=1;for(ui i=1;i<len;++i) ans=(ans*(MOD+1)+1);return ans;}
				ui P,P2,NP,Pk;static constexpr ui ui_len = sizeof(ui)*8;
				montgomery_mi_lib(ui P0):P(P0),P2(P0*2),NP((-ull(P0))%P0),Pk(calc_k(P0,ui_len)){}
				montgomery_mi_lib(){}
				#define INLINE_OP __attribute__((__always_inline__))
				INLINE_OP ui redd(ui k) const {return k>=P2?k-P2:k;}INLINE_OP ui reds(ui k) const {return k>=P?k-P:k;}INLINE_OP ui redu(ull k) const {return (k+ull(ui(k)*Pk)*P)>>ui_len;}
				INLINE_OP ui add(ui a,ui b) const {return redd(a+b);}INLINE_OP ui sub(ui a,ui b) const {return redd(a-b+P2);}INLINE_OP ui mul(ui a,ui b) const {return redu(ull(a)*b);}
				INLINE_OP ui neg(ui a) const {return redd(P2-a);}INLINE_OP ui v(ui a) const {return redu(ull(a)*NP);}INLINE_OP ui rv(ui a) const {return reds(redu(a));}
				#undef INLINE_OP
			};
			struct montgomery_mi{
				static montgomery_mi_lib mlib;
				#if defined(_OPENMP)
				#pragma omp threadprivate(mlib)
				#endif
				ui val;
				void init(ui a){val=mlib.redu(ull(a)*mlib.NP);}
				montgomery_mi(){val=0;} montgomery_mi(const montgomery_mi &a):val(a.val){} montgomery_mi(ui v):val(mlib.redu(ull(v)*mlib.NP)){}
				montgomery_mi& operator=(const montgomery_mi &b) {val=b.val;return *this;}
				montgomery_mi& operator+=(const montgomery_mi &b) {val=mlib.redd(val+b.val);return *this;}
				montgomery_mi& operator-=(const montgomery_mi &b) {val=mlib.redd(val-b.val+mlib.P2);return *this;}
				montgomery_mi operator+(const montgomery_mi &b) const {return montgomery_mi(*this)+=b;}
				montgomery_mi operator-(const montgomery_mi &b) const {return montgomery_mi(*this)-=b;}
				montgomery_mi operator*=(const montgomery_mi &b) {val=mlib.redu(ull(val)*b.val);return *this;}
				montgomery_mi operator*(const montgomery_mi &b) const {return montgomery_mi(*this)*=b;}
				montgomery_mi operator-() const {montgomery_mi b;return b-=(*this);}
				ui real_val() const {return mlib.reds(mlib.redu(val));}
				ui get_val() const {return val;}
				friend std::istream& operator>>(std::istream &in, montgomery_mi &m_int){ui inp;in>>inp;m_int.init(inp);return in;}
				friend std::ostream& operator<<(std::ostream &out, const montgomery_mi &m_int){out<<m_int.real_val();return out;}
			};
			struct montgomery_mli_lib{
				constexpr static ull calc_k(ull MOD,ui len){ull ans=1;for(ui i=1;i<len;++i) ans=(ans*(MOD+1)+1);return ans;}
				ull P,P2,NP,Pk;static constexpr ui ull_len = sizeof(ull)*8;
				montgomery_mli_lib(ull P0):P(P0),P2(P0*2),NP((-u128(P0))%P0),Pk(calc_k(P0,ull_len)){}
				montgomery_mli_lib(){}
				ull redd(ull k) const {return k>=P2?k-P2:k;}ull reds(ull k) const {return k>=P?k-P:k;}ull redu(u128 k) const {return (k+u128(ull(k)*Pk)*P)>>ull_len;}
			};
			struct montgomery_mli{
				static montgomery_mli_lib mlib;
				#if defined(_OPENMP)
				#pragma omp threadprivate(mlib)
				#endif
				ull val;
				void init(ull a){val=mlib.redu(ull(a)*mlib.NP);}
				montgomery_mli(){val=0;} montgomery_mli(const montgomery_mli &a):val(a.val){} montgomery_mli(ull v):val(mlib.redu(u128(v)*mlib.NP)){}
				montgomery_mli& operator=(const montgomery_mli &b) {val=b.val;return *this;}
				montgomery_mli& operator+=(const montgomery_mli &b) {val=mlib.redd(val+b.val);return *this;}
				montgomery_mli& operator-=(const montgomery_mli &b) {val=mlib.redd(val-b.val+mlib.P2);return *this;}
				montgomery_mli operator+(const montgomery_mli &b) const {return montgomery_mli(*this)+=b;}
				montgomery_mli operator-(const montgomery_mli &b) const {return montgomery_mli(*this)-=b;}
				montgomery_mli operator*=(const montgomery_mli &b) {val=mlib.redu(u128(val)*b.val);return *this;}
				montgomery_mli operator*(const montgomery_mli &b) const {return montgomery_mli(*this)*=b;}
				montgomery_mli operator-() const {montgomery_mli b;return b-=(*this);}
				ull real_val() const {return mlib.reds(mlib.redu(val));}
				ull get_val() const {return val;}
				friend std::istream& operator>>(std::istream &in, montgomery_mli &m_int){ull inp;in>>inp;m_int.init(inp);return in;}
				friend std::ostream& operator<<(std::ostream &out, const montgomery_mli &m_int){out<<m_int.real_val();return out;}
			};
			typedef montgomery_mi mi;
			typedef montgomery_mli mli;
			typedef montgomery_mi_lib lmi;
			#if defined(__AVX__) && defined(__AVX2__)
			struct montgomery_mm256_lib{
				alignas(32) __m256i P,P2,NP,Pk;static constexpr ui ui_len=sizeof(ui)*8;
				montgomery_mm256_lib(ui P0){P=_mm256_set1_epi32(P0),P2=_mm256_set1_epi32(P0*2),
					NP=_mm256_set1_epi32(ui((-ull(P0))%P0)),Pk=_mm256_set1_epi32(montgomery_mi_lib::calc_k(P0,ui_len));}
				montgomery_mm256_lib(){}
				#define INLINE_OP __attribute__((__always_inline__))
				INLINE_OP __m256i redd(__m256i k){__m256i a=_mm256_sub_epi32(k,P2);__m256i b=_mm256_and_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(),a),P2);return _mm256_add_epi32(a,b);}
				INLINE_OP __m256i reds(__m256i k){__m256i a=_mm256_sub_epi32(k,P); __m256i b=_mm256_and_si256(_mm256_cmpgt_epi32(_mm256_setzero_si256(),a),P); return _mm256_add_epi32(a,b);}
				INLINE_OP __m256i redu(__m256i k){return _mm256_srli_epi64(_mm256_add_epi64(_mm256_mul_epu32(_mm256_mul_epu32(k,Pk),P),k),32);}
				INLINE_OP __m256i mul(__m256i k1,__m256i k2){
					return _mm256_or_si256(redu(_mm256_mul_epu32(k1,k2)),_mm256_slli_epi64(redu(_mm256_mul_epu32(_mm256_srli_epi64(k1,32),_mm256_srli_epi64(k2,32))),32));}
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
				montgomery_mm256_int(){val=_mm256_setzero_si256();} montgomery_mm256_int(const montgomery_mm256_int &a):val(a.val){} montgomery_mm256_int(__m256i v):val(mlib.mul(v,mlib.NP)){}
				montgomery_mm256_int(ui v){init(_mm256_set1_epi32(v));}
				montgomery_mm256_int& operator=(const montgomery_mm256_int &b) {val=b.val;return *this;}
				montgomery_mm256_int& operator+=(const montgomery_mm256_int &b) {val=mlib.add(val,b.val);return *this;}
				montgomery_mm256_int& operator-=(const montgomery_mm256_int &b) {val=mlib.sub(val,b.val);return *this;}
				montgomery_mm256_int operator+(const montgomery_mm256_int &b) const {return montgomery_mm256_int(*this)+=b;}
				montgomery_mm256_int operator-(const montgomery_mm256_int &b) const {return montgomery_mm256_int(*this)-=b;}
				montgomery_mm256_int operator*=(const montgomery_mm256_int &b) {val=mlib.mul(val,b.val);return *this;}
				montgomery_mm256_int operator*(const montgomery_mm256_int &b) const {return montgomery_mm256_int(*this)*=b;}
				montgomery_mm256_int operator-() const {montgomery_mm256_int b;return b-=(*this);}
				__m256i real_val() const {return mlib.reds(_mm256_or_si256(_mm256_slli_epi64(mlib.redu(_mm256_srli_epi64(val,32)),32),
					mlib.redu(_mm256_srli_epi64(_mm256_slli_epi64(val,32),32))));}
				__m256i get_val() const {return val;}
			};
			typedef montgomery_mm256_int mai;
			typedef montgomery_mm256_lib lma;
			#endif
			#if defined(__AVX512F__) && defined(__AVX512DQ__)
			struct montgomery_mm512_lib{
				alignas(64) __m512i P,P2,NP,Pk;static constexpr ui ui_len=sizeof(ui)*8;
				montgomery_mm512_lib(ui P0){P=_mm512_set1_epi32(P0),P2=_mm512_set1_epi32(P0*2),
					NP=_mm512_set1_epi32(ui((-ull(P0))%P0)),Pk=_mm512_set1_epi32(montgomery_mi_lib::calc_k(P0,ui_len));}
				montgomery_mm512_lib(){}
				#define INLINE_OP __attribute__((__always_inline__))
				INLINE_OP __m512i redd(__m512i k){__m512i a=_mm512_sub_epi32(k,P2);return _mm512_mask_add_epi32(a,_mm512_cmpgt_epi32_mask(_mm512_setzero_si512(),a),a,P2);}
				INLINE_OP __m512i reds(__m512i k){__m512i a=_mm512_sub_epi32(k,P);return _mm512_mask_add_epi32(a,_mm512_cmpgt_epi32_mask(_mm512_setzero_si512(),a),a,P);}
				INLINE_OP __m512i redu(__m512i k){return _mm512_srli_epi64(_mm512_add_epi64(_mm512_mul_epu32(_mm512_mul_epu32(k,Pk),P),k),32);}
				INLINE_OP __m512i mul(__m512i k1,__m512i k2){
					return _mm512_or_si512(redu(_mm512_mul_epu32(k1,k2)),_mm512_slli_epi64(redu(_mm512_mul_epu32(_mm512_srli_epi64(k1,32),_mm512_srli_epi64(k2,32))),32));}
				INLINE_OP __m512i add(__m512i k1,__m512i k2){return redd(_mm512_add_epi32(k1,k2));}
				INLINE_OP __m512i sub(__m512i k1,__m512i k2){return redd(_mm512_add_epi32(P2,_mm512_sub_epi32(k1,k2)));}
				#undef INLINE_OP
			};
			struct montgomery_mm512_int{
				static montgomery_mm512_lib mlib;
				#if defined(_OPENMP)
				#pragma omp threadprivate(mlib)
				#endif
				__m512i val;
				void init(__m512i a){val=mlib.mul(a,mlib.NP);}
				montgomery_mm512_int(){val=_mm512_setzero_si512();} montgomery_mm512_int(const montgomery_mm512_int &a):val(a.val){} montgomery_mm512_int(__m512i v):val(mlib.mul(v,mlib.NP)){}
				montgomery_mm512_int(ui v){init(_mm512_set1_epi32(v));}
				montgomery_mm512_int& operator=(const montgomery_mm512_int &b) {val=b.val;return *this;}
				montgomery_mm512_int& operator+=(const montgomery_mm512_int &b) {val=mlib.add(val,b.val);return *this;}
				montgomery_mm512_int& operator-=(const montgomery_mm512_int &b) {val=mlib.sub(val,b.val);return *this;}
				montgomery_mm512_int operator+(const montgomery_mm512_int &b) const {return montgomery_mm512_int(*this)+=b;}
				montgomery_mm512_int operator-(const montgomery_mm512_int &b) const {return montgomery_mm512_int(*this)-=b;}
				montgomery_mm512_int operator*=(const montgomery_mm512_int &b) {val=mlib.mul(val,b.val);return *this;}
				montgomery_mm512_int operator*(const montgomery_mm512_int &b) const {return montgomery_mm512_int(*this)*=b;}
				montgomery_mm512_int operator-() const {montgomery_mm512_int b;return b-=(*this);}
				__m512i real_val() const {return mlib.reds(_mm512_or_si512(_mm512_slli_epi64(mlib.redu(_mm512_srli_epi64(val,32)),32),
					mlib.redu(_mm512_srli_epi64(_mm512_slli_epi64(val,32),32))));}
				__m512i get_val() const {return val;}
			};
			typedef montgomery_mm512_int m5i;
			typedef montgomery_mm512_lib lm5;
			#endif
			extern ui global_mod_mi;
			extern ull global_mod_mli;
			#if defined(_OPENMP)
			#pragma omp threadprivate(global_mod_mi)
			#pragma omp threadprivate(global_mod_mli)
			#endif
			void set_mod_mi(ui p);
			void set_mod_mli(ull p);
			void set_mod_for_all_threads_mi(ui p);
			void set_mod_for_all_threads_mli(ull p);
			#if defined(__AVX__) && defined(__AVX2__)
			extern ui global_mod_mai;
			#if defined(_OPENMP)
			#pragma omp threadprivate(global_mod_mai)
			#endif
			void set_mod_mai(ui p);
			void set_mod_for_all_threads_mai(ui p);
			#endif
			#if defined(__AVX512F__) && defined(__AVX512DQ__)
			extern ui global_mod_m5i;
			#if defined(_OPENMP)
			#pragma omp threadprivate(global_mod_m5i)
			#endif
			void set_mod_m5i(ui p);
			void set_mod_for_all_threads_m5i(ui p);
			#endif
		}
	}
	using modulo::mod_int::mi;
	using modulo::mod_int::mli;
	using modulo::mod_int::set_mod_mi;
	using modulo::mod_int::set_mod_mli;
	using modulo::mod_int::global_mod_mi;
	using modulo::mod_int::global_mod_mli;
	using modulo::mod_int::default_mod;
	using modulo::mod_int::lmi;
	#if defined(__AVX__) && defined(__AVX2__)
	using modulo::mod_int::mai;
	using modulo::mod_int::global_mod_mai;
	using modulo::mod_int::set_mod_mai;
	using modulo::mod_int::lma;
	#endif
	#if defined(__AVX512F__) && defined(__AVX512DQ__)
	using modulo::mod_int::m5i;
	using modulo::mod_int::lm5;
	#endif
	namespace traits
	{
		template<typename X, typename Y,typename T,typename Op>
		struct op_valid_helper
		{
			template<typename U, typename L, typename R,typename G>
			static auto test(int) -> std::is_same<std::remove_cv_t<std::remove_reference_t<
			decltype(std::declval<U>()(std::declval<std::add_lvalue_reference_t<std::remove_cv_t<std::remove_reference_t<R>>>>(),
					 std::declval<std::add_lvalue_reference_t<std::remove_cv_t<std::remove_reference_t<R>>>>()))>>,
			std::remove_cv_t<std::remove_reference_t<G>>>;
			template<typename U, typename L, typename R,typename G>
			static auto test(...) -> std::false_type;
			using type = decltype(test<Op, X, Y,T>(0));
		};
		template<typename X, typename Y,typename W,typename Op> using op_valid = typename op_valid_helper<X,Y,W,Op>::type;
		template<typename X, typename Y,typename W,typename Op> constexpr bool op_valid_v=op_valid<X,Y,W,Op>::value;
		template<typename X, typename Y,typename T,typename Op>
		struct const_op_valid_helper
		{
			template<typename U, typename L, typename R,typename G>
			static auto test(int) -> std::is_same<std::remove_cv_t<std::remove_reference_t<
			decltype(std::declval<U>()(std::declval<std::add_lvalue_reference_t<std::add_const_t<std::remove_cv_t<std::remove_reference_t<L>>>>>(),
					 std::declval<std::add_lvalue_reference_t<std::add_const_t<std::remove_cv_t<std::remove_reference_t<R>>>>>()))>>,
			std::remove_cv_t<std::remove_reference_t<G>>>;
			template<typename U, typename L, typename R,typename G>
			static auto test(...) -> std::false_type;
			using type = decltype(test<Op, X, Y,T>(0));
		};
		template<typename X, typename Y,typename W,typename Op> using const_op_valid = typename const_op_valid_helper<X,Y,W,Op>::type;
		template<typename X, typename Y,typename W,typename Op> constexpr bool const_op_valid_v=const_op_valid<X,Y,W,Op>::value;
		template<typename T> struct is_mod_int:std::false_type{};
		template<> struct is_mod_int<mi>:std::true_type{};
		template<> struct is_mod_int<mli>:std::true_type{};
		#if defined(__AVX__) && defined(__AVX2__)
		template<> struct is_mod_int<mai>:std::true_type{};
		#endif
		#if defined(__AVX512F__) && defined(__AVX512DQ__)
		template<> struct is_mod_int<m5i>:std::true_type{};
		#endif
		template<typename T> constexpr bool is_mod_int_v=is_mod_int<T>::value;
		template<typename T>
		struct has_unit_value_helper
		{
			template<typename G> static auto test(int) -> typename std::enable_if<(std::is_arithmetic_v<G> || is_mod_int_v<G>),std::true_type>::type;
			template<typename G> static auto test(...) -> std::false_type;
			using type=decltype(test<T>(0));
		};
		template<typename T> using has_unit_value=typename has_unit_value_helper<T>::type;
		template<typename T> constexpr bool has_unit_value_v=has_unit_value<T>::value;
		template<typename T,typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
		T unit_value(){return 1;}
		template<typename T,typename std::enable_if<is_mod_int<T>::value>::type* = nullptr>
		T unit_value(){return T(1);}
		namespace integer_helper
		{
			template<typename T> struct type_helper{typedef T type;};
			template<size_t N> struct size_int:size_int<N+1>{};
			template<> struct size_int<8>:type_helper<int8_t>{};
			template<> struct size_int<16>:type_helper<int16_t>{};
			template<> struct size_int<32>:type_helper<int32_t>{};
			template<> struct size_int<64>:type_helper<int64_t>{};
			template<> struct size_int<128>:type_helper<__int128>{};
			template<size_t N> using size_int_t=typename size_int<N>::type;
			template<size_t N> struct size_uint:size_uint<N+1>{};
			template<> struct size_uint<8>:type_helper<uint8_t>{};
			template<> struct size_uint<16>:type_helper<uint16_t>{};
			template<> struct size_uint<32>:type_helper<uint32_t>{};
			template<> struct size_uint<64>:type_helper<uint64_t>{};
			template<> struct size_uint<128>:type_helper<__uint128_t>{};
			template<size_t N> using size_uint_t=typename size_uint<N>::type;
			template<typename T,typename std::enable_if<std::is_integral<T>::value>::type* =nullptr> struct double_width_int:size_int<2*sizeof(T)*8>{};
			template<typename T> using double_width_int_t=typename double_width_int<T>::type;
			template<typename T,typename std::enable_if<std::is_integral<T>::value>::type* =nullptr> struct double_width_uint:size_uint<2*sizeof(T)*8>{};
			template<typename T> using double_width_uint_t=typename double_width_uint<T>::type;
		}
	}
	namespace basic
	{
		template<typename B,typename U>
		typename std::enable_if<(traits::const_op_valid_v<B,B,B,std::multiplies<>> &&
								 std::is_move_assignable_v<std::remove_cv_t<std::remove_reference_t<B>>> &&
								 traits::has_unit_value_v<std::remove_cv_t<std::remove_reference_t<B>>> &&
								 std::is_integral_v<std::remove_cv_t<std::remove_reference_t<U>>>),
								 std::remove_cv_t<std::remove_reference_t<B>>>::type
		fast_pow(B b,U u){
			typedef typename std::make_unsigned_t<std::remove_cv_t<std::remove_reference_t<U>>> UU;
			typedef typename std::remove_cv_t<std::remove_reference_t<B>> BB;
			UU u0=static_cast<UU>(u);BB ans=traits::unit_value<BB>(),off=b;
			while(u0){if(u0&1) ans=ans*off;off=off*off;u0>>=1;}return ans;
		}
		template<typename B,typename U,typename M>
		typename std::enable_if<(std::is_integral_v<std::remove_cv_t<std::remove_reference_t<B>>> &&
								 std::is_integral_v<std::remove_cv_t<std::remove_reference_t<U>>> &&
								 std::is_integral_v<std::remove_cv_t<std::remove_reference_t<M>>>),
								 traits::integer_helper::size_uint_t<sizeof(std::remove_cv_t<std::remove_reference_t<M>>)*8>>::type
		fast_pow(B b,U u,M m){
			typedef typename std::make_unsigned_t<std::remove_cv_t<std::remove_reference_t<U>>> UU;
			typedef typename traits::integer_helper::double_width_uint_t<std::remove_cv_t<std::remove_reference_t<M>>> MM;
			typedef typename traits::integer_helper::size_uint_t<sizeof(std::remove_cv_t<std::remove_reference_t<M>>)*8> M0;
			UU u0=static_cast<UU>(u);M0 ans=1,off=static_cast<M0>(b%m),md=static_cast<M0>(m);
			while(u0){if(u0&1) ans=(MM)ans*off%md;off=(MM)off*off%md;u0>>=1;}return ans;
		}
	}
	namespace power_series_ring
	{
		typedef std::vector<mi> poly;
		namespace polynomial_kernel
		{
			class polynomial_kernel;
			class polynomial_kernel_ntt
			{
			private:
				static constexpr ui tmp_size=9;
				aligned_array<ui,32> ws0,ws1,_inv,tt[tmp_size],num;ui P,G;
				ui fn,fb,mx;
				void release();
				ui _fastpow(ui a,ui b);
				void dif(ui* restrict arr,ui n);
				void dit(ui* restrict arr,ui n);
				void dif_xni(ui* restrict arr,ui n);
				void dit_xni(ui* restrict arr,ui n);
				void internal_mul(ui* restrict src1,ui* restrict src2,ui* restrict dst,ui m);
				void internal_inv(ui* restrict src,ui* restrict dst,ui* restrict tmp,ui* restrict tmp2,ui len);
				void internal_inv_faster(ui* restrict src,ui* restrict dst,ui* restrict tmp,ui* restrict tmp2,ui* restrict tmp3,ui len);
				void internal_ln(ui* restrict src,ui* restrict dst,ui* restrict tmp1,ui* restrict tmp2,ui* restrict tmp3,ui len);
				void internal_ln_faster(ui* restrict src,ui* restrict dst,ui* restrict tmp,ui* restrict tmp2,ui* restrict tmp3,ui* restrict tmp4,ui len);
				void internal_exp(ui* restrict src,ui* restrict dst,ui* restrict gn,ui* restrict gxni,
								  ui* restrict h,ui* restrict tmp1,ui* restrict tmp2,ui* restrict tmp3,ui len,bool calc_h=false);
				lmi li;
				#if defined(__AVX__) && defined(__AVX2__)
				lma la;
				#endif
			public:
				friend class polynomial_kernel;
				polynomial_kernel_ntt(ui max_conv_size,ui P0,ui G0);
				void init(ui max_conv_size,ui P0,ui G0);
				polynomial_kernel_ntt(const polynomial_kernel_ntt &d);
				polynomial_kernel_ntt();
				~polynomial_kernel_ntt();
				poly mul(const poly &a,const poly &b);
				poly inv(const poly &src);
				poly ln(const poly &src);
				poly exp(const poly &src);
				std::array<long long,7> test(ui T);
			};
		}
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