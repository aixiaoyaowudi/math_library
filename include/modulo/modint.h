/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_MODULO_MODINT_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_MODULO_MODINT_H_

#include <type/basic_typedef.h>
#include <immintrin.h>
#include <iostream>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace math
{
	namespace modulo
	{
		namespace modint
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
			static __attribute__((__always_inline__)) inline montgomery_mi ui2mi(ui k){
				union ui2mi_impl
				{
					ui uval;
					montgomery_mi mval;
					ui2mi_impl(){}
				}T;
				T.uval=k;
				return T.mval;
			}
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
			static __attribute__((__always_inline__)) inline montgomery_mli ull2mli(ull k){
				union ull2mli_impl
				{
					ull uval;
					montgomery_mli mval;
					ull2mli_impl(){}
				}T;
				T.uval=k;
				return T.mval;
			}
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
				INLINE_OP __m512i redd(__m512i k){__m512i a=_mm512_sub_epi32(k,P2);return _mm512_mask_add_epi32(a,_mm512_movepi32_mask(a),a,P2);}
				INLINE_OP __m512i reds(__m512i k){__m512i a=_mm512_sub_epi32(k,P);return _mm512_mask_add_epi32(a,_mm512_movepi32_mask(a),a,P);}
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
	using modulo::modint::mi;
	using modulo::modint::mli;
	using modulo::modint::set_mod_mi;
	using modulo::modint::set_mod_mli;
	using modulo::modint::global_mod_mi;
	using modulo::modint::global_mod_mli;
	using modulo::modint::default_mod;
	using modulo::modint::lmi;
	using modulo::modint::ui2mi;
	using modulo::modint::ull2mli;
	#if defined(__AVX__) && defined(__AVX2__)
	using modulo::modint::mai;
	using modulo::modint::global_mod_mai;
	using modulo::modint::set_mod_mai;
	using modulo::modint::lma;
	#endif
	#if defined(__AVX512F__) && defined(__AVX512DQ__)
	using modulo::modint::m5i;
	using modulo::modint::lm5;
	#endif
}

#endif