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

#if defined(__AVX__) && defined(__AVX2__)
#pragma message "AVX & AVX2 acceleration enabled."
#endif

#if defined(_OPENMP)
#pragma message "Openmp acceleration enabled."
#endif

namespace math
{
	typedef uint32_t uint;
	typedef int64_t ll;
	typedef uint64_t ull;
	typedef __uint128_t L;
	struct FastMod {
		ull b, m;
		FastMod(ull b) : b(b), m(ull((L(1) << 64) / b)) {}
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
	decltype(mint::mlib) mint::mlib=montgomery_int_lib(default_mod);
	static void set_mint_mod(uint32_t p){
		mint::mlib=montgomery_int_lib(p);
	}
	mint fast_pow(mint a,ull b){mint ans=mint(1),off=a;while(b){if(b&1) ans*=off;off*=off;b>>=1;}return ans;}
	#if defined(__AVX__) && defined(__AVX2__)
	struct montgomery_mm256_lib{
		alignas(32) __m256i P,P2,NP,Pk,mask1;
		montgomery_mm256_lib(uint P0){P=_mm256_set1_epi32(P0),P2=_mm256_set1_epi32(P0*2),mask1=_mm256_setr_epi32(0,P0*2,0,P0*2,0,P0*2,0,P0*2),
			NP=_mm256_set1_epi32(uint((-ull(P0))%P0)),Pk=_mm256_set1_epi32(montgomery_int_lib::calc_k(P0,sizeof(uint)*8));}
		montgomery_mm256_lib(){}
		#define INLINE_OP __attribute__((always_inline))
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
	decltype(m256int::mlib) m256int::mlib=montgomery_mm256_lib(default_mod);
	static void set_m256int_mod(uint32_t p){
		m256int::mlib=montgomery_mm256_lib(p);
	}
	#endif
	uint32_t global_mod=default_mod;
	#if defined(_OPENMP)
	#pragma omp threadprivate(global_mod)
	#endif
	void set_mod_for_all_threads(uint32_t p){
		#if defined(_OPENMP)
		#pragma omp parallel
		{
		#endif
			set_mint_mod(p);
			global_mod=p;
			#if defined(__AVX__) && defined(__AVX2__)
			set_m256int_mod(p);
			#endif
		#if defined(_OPENMP)
		}
		#endif
	}
	void set_mod(uint32_t p){
		set_mint_mod(p);
		global_mod=p;
		#if defined(__AVX__) && defined(__AVX2__)
		set_m256int_mod(p);
		#endif
	}
	typedef std::vector<mint> poly;
	template<uint32_t P> class polynomial;
	template<uint32_t P,uint32_t G>
	class polynomial_ntt
	{
	private:
		mint *ws0,*ws1,*fac,*ifac,*_inv,*tt;uint fn,fb,*rev,*lgg;
		bool alloced;
		void release(){
			if(alloced){
				operator delete[](ws0, std::align_val_t(32));
				operator delete[](ws1, std::align_val_t(32));
				operator delete[](fac, std::align_val_t(32));
				operator delete[](ifac,std::align_val_t(32));
				operator delete[](_inv,std::align_val_t(32));
				operator delete[](tt,  std::align_val_t(32));
				operator delete[](rev, std::align_val_t(32));
				operator delete[](lgg, std::align_val_t(32));
				alloced=false;
			}
		}
	public:
		template<uint32_t P0>
		friend class polynomial;
		polynomial_ntt(){alloced=false;ws0=ws1=fac=ifac=_inv=tt=NULL;rev=lgg=NULL;fn=fb=0;}
		void init(uint max_conv_size){
			release();
			fn=1;fb=0;while(fn<(max_conv_size<<1)) fn<<=1,++fb;
			_inv=new(std::align_val_t(32)) mint[fn+32];
			ws0 =new(std::align_val_t(32)) mint[fn+32];
			ws1 =new(std::align_val_t(32)) mint[fn+32];
			fac =new(std::align_val_t(32)) mint[fn+32];
			ifac=new(std::align_val_t(32)) mint[fn+32];
			rev =new(std::align_val_t(32)) uint[fn+32];
			lgg =new(std::align_val_t(32)) uint[fn+32];
			tt  =new(std::align_val_t(32)) mint[fn+32];
			alloced=true;
			_inv[1]=mint(1);for(uint i=2;i<=fn;++i) _inv[i]=(-mint(P/i))*_inv[P%i];
			ifac[0]=fac[0]=mint(1);for(uint i=1;i<=fn;++i) fac[i]=mint(i)*fac[i-1],ifac[i]=ifac[i-1]*_inv[i];
			rev[0]=0;for(uint i=0;i<fn;++i) rev[i]=(rev[i>>1]>>1)|((i&1)<<(fb-1));
			lgg[0]=lgg[1]=0;for(uint i=2;i<=fn;++i) lgg[i]=lgg[(i+1)>>1]+1;
			mint j0=fast_pow(mint(G),(P-1)/fn),j1=fast_pow(fast_pow(mint(G),(P-2)),(P-1)/fn);
			for(uint mid=(fn>>1);mid>=1;mid>>=1,j0*=j0,j1*=j1){
				mint w0(1),w1(1);
				for(uint i=0;i<mid;++i,w0*=j0,w1*=j1) ws0[i+mid]=w0,ws1[i+mid]=w1;
			}
		}
		polynomial_ntt(const polynomial_ntt<P,G> &d){
			alloced=d.alloced;fn=d.fn,fb=d.fb;
			if(alloced){
				_inv=new(std::align_val_t(32)) mint[fn+32];
				ws0 =new(std::align_val_t(32)) mint[fn+32];
				ws1 =new(std::align_val_t(32)) mint[fn+32];
				fac =new(std::align_val_t(32)) mint[fn+32];
				ifac=new(std::align_val_t(32)) mint[fn+32];
				rev =new(std::align_val_t(32)) uint[fn+32];
				lgg =new(std::align_val_t(32)) uint[fn+32];
				tt  =new(std::align_val_t(32)) mint[fn+32];
				std::memcpy(ws0, d.ws0, sizeof(mint)*(fn+32));
				std::memcpy(ws1, d.ws1, sizeof(mint)*(fn+32));
				std::memcpy(fac, d.fac, sizeof(mint)*(fn+32));
				std::memcpy(ifac,d.ifac,sizeof(mint)*(fn+32));
				std::memcpy(_inv,d._inv,sizeof(mint)*(fn+32));
				std::memcpy(tt,  d.tt,  sizeof(mint)*(fn+32));
				std::memcpy(rev, d.rev, sizeof(uint)*(fn+32));
				std::memcpy(lgg, d.lgg, sizeof(uint)*(fn+32));
			}
		}
		// TODO: add avx support
		// #if defined(__AVX__) && defined(__AVX2)
		// #else
		void NTT(poly &p,int V){
			uint bts=lgg[p.size()];if(p.size()!=(1<<bts)) p.resize((1<<bts));
			mint *w=(V==1)?ws0:ws1;uint len=(1<<bts);for(uint i=0;i<len;++i) tt[i]=p[rev[i]>>(fb-bts)];
			mint t1,t2;
			for(uint l=2;l<=len;l<<=1)
				for(uint j=0,mid=(l>>1);j<len;j+=l)
					for(uint i=0;i<mid;++i) t1=tt[j+i],t2=tt[j+i+mid]*w[mid+i],tt[j+i]=t1+t2,tt[j+i+mid]=t1-t2;
			if(V==1) for(uint i=0;i<len;++i) p[i]=tt[i];
			else{mint j=_inv[len];for(uint i=0;i<len;++i) p[i]=tt[i]*j;}
		}
		poly mul(const poly &a,const poly &b){
			poly p1(a),p2(b);uint len=a.size()+b.size()-1,ff=(1<<lgg[len]);
			p1.resize(ff),p2.resize(ff);NTT(p1,1);NTT(p2,1);
			for(uint i=0;i<ff;++i) p1[i]*=p2[i];NTT(p1,-1);
			p1.resize(len);return p1;
		}
		poly inv(const poly &a){
			uint l=a.size();if(l==1){poly ret(1);ret[0]=fast_pow(mint(a[0]),P-2);return ret;}
			poly g0=a;g0.resize((l+1)>>1);g0=inv(g0);
			poly p1(a);uint ff=(2<<lgg[l]);g0.resize(ff);p1.resize(ff);mint m2(2);
			NTT(p1,1);NTT(g0,1);for(uint i=0;i<ff;++i) g0[i]=g0[i]*(m2-g0[i]*p1[i]);
			NTT(g0,-1);g0.resize(l);return g0;
		}
		poly ln(const poly &a){
			uint l=a.size();poly p1(l-1);
			for(uint i=1;i<l;++i) p1[i-1]=mint(i)*a[i];
			p1=mul(p1,inv(a));p1.resize(l-1);poly ret(l);
			for(uint i=1;i<l;++i) ret[i]=_inv[i]*p1[i-1];ret[0]=mint(0);
			return ret;
		}
		poly exp(const poly &a){
			uint l=a.size();if(l==1){poly ret(1);ret[0]=mint(1);return ret;}
			poly g0=a;g0.resize((l+1)>>1);g0=exp(g0);poly g1(g0),g2(a);g1.resize(l);g1=ln(g1);
			uint ff=(2<<lgg[l]);g0.resize(ff);g1.resize(ff);g2.resize(ff);
			NTT(g0,1);NTT(g1,1);NTT(g2,1);mint m1(1);
			for(uint i=0;i<ff;++i) g0[i]=g0[i]*(m1-g1[i]+g2[i]);
			NTT(g0,-1);g0.resize(l);return g0;
		}
		// #endif
		~polynomial_ntt(){release();}
	};
	template<uint32_t P>
	class polynomial
	{
	private:
		static constexpr uint P1=469762049,P2=998244353,P3=1004535809,I1=554580198,I2=395249030;
		polynomial_ntt<P1,3> pn1;
		polynomial_ntt<P2,3> pn2;
		polynomial_ntt<P3,3> pn3;
		FastMod F,F1,F2,F3;uint N3;
		mint *_inv;bool alloced;
	public:
		void release(){
			if(alloced){
				alloced=false;
				pn1.release();
				pn2.release();
				pn3.release();
				operator delete[](_inv,std::align_val_t(32));
			}
		}
		polynomial(const polynomial<P> &d):pn1(d.pn1),pn2(d.pn2),pn3(d.pn3),F(P),F1(P1),F2(P2),F3(P3),N3(1ull*P1*P2%P){
			alloced=d.alloced;
			if(alloced){
				_inv=new(std::align_val_t(32)) mint[pn1.fn+32];
				memcpy(_inv,d._inv,sizeof(mint)*(d.pn1.fn+32));
			}
		}
		polynomial():F(P),F1(P2),F2(P2),F3(P3),N3(1ull*P1*P2%P){alloced=false;}
		~polynomial(){release();}
		void init(uint max_conv_size){
			release();
			alloced=true;
			set_mod(P1);
			pn1.init(max_conv_size);
			set_mod(P2);
			pn2.init(max_conv_size);
			set_mod(P3);
			pn3.init(max_conv_size);
			set_mod(P);
			_inv=new(std::align_val_t(32)) mint[pn1.fn+32];_inv[1]=mint(1);
			for(uint i=2;i<=pn1.fn;++i) _inv[i]=(-mint(P/i))*_inv[P%i];
		}
		poly mul(const poly &a,const poly &b){
			uint la=a.size(),lb=b.size();uint len=a.size()+b.size()-1,ff=(1<<(pn1.lgg[len]));std::vector<uint> r1(len),r2(len),r3(len),na(la),nb(lb);poly ret(len),p1(ff),p2(ff);
			for(uint i=0;i<la;++i) na[i]=a[i].real_val();
			for(uint i=0;i<lb;++i) nb[i]=b[i].real_val();
			set_mod(P1);
			for(uint i=0;i<la;++i) p1[i]=mint(na[i]);
			for(uint i=0;i<lb;++i) p2[i]=mint(nb[i]);
			pn1.NTT(p1,1);pn1.NTT(p2,1);
			for(uint i=0;i<ff;++i) p1[i]*=p2[i];
			pn1.NTT(p1,-1);
			for(uint i=0;i<len;++i) r1[i]=p1[i].real_val();
			set_mod(P2);
			for(uint i=0;i<la;++i) p1[i]=mint(na[i]);for(uint i=la;i<ff;++i) p1[i]=mint(0);
			for(uint i=0;i<lb;++i) p2[i]=mint(nb[i]);for(uint i=lb;i<ff;++i) p2[i]=mint(0);
			pn2.NTT(p1,1);pn2.NTT(p2,1);
			for(uint i=0;i<ff;++i) p1[i]*=p2[i];
			pn2.NTT(p1,-1);
			for(uint i=0;i<len;++i) r2[i]=p1[i].real_val();
			set_mod(P3);
			for(uint i=0;i<la;++i) p1[i]=mint(na[i]);for(uint i=la;i<ff;++i) p1[i]=mint(0);
			for(uint i=0;i<lb;++i) p2[i]=mint(nb[i]);for(uint i=lb;i<ff;++i) p2[i]=mint(0);
			pn3.NTT(p1,1);pn3.NTT(p2,1);
			for(uint i=0;i<ff;++i) p1[i]*=p2[i];
			pn3.NTT(p1,-1);
			for(uint i=0;i<len;++i) r3[i]=p1[i].real_val();
			set_mod(P);
			for(uint i=0;i<len;++i){
				uint k1=F2.reduce(1ull*I1*(r2[i]-r1[i]+P2));
				ull x4=r1[i]+1ull*k1*P1;uint k4=F3.reduce((r3[i]-F3.reduce(x4)+P3)*I2);
				ret[i]=mint(F.reduce(1ull*N3*k4+x4));
			}
			return ret;
		}
		poly inv(const poly &a){
			uint l=a.size();if(l==1){poly ret(1);ret[0]=fast_pow(mint(a[0]),P-2);return ret;}
			poly g0=a;g0.resize((l+1)>>1);g0=inv(g0);poly g1=mul(a,g0);g1.resize(l);
			for(uint i=0;i<l;++i) g1[i]=-g1[i];g1[0]+=mint(2);g1=mul(g1,g0);g1.resize(l);
			return g1;
		}
		poly ln(const poly &a){
			uint l=a.size();poly p1(l-1);
			for(uint i=1;i<l;++i) p1[i-1]=mint(i)*a[i];
			p1=mul(p1,inv(a));p1.resize(l-1);poly ret(l);
			for(uint i=1;i<l;++i) ret[i]=_inv[i]*p1[i-1];ret[0]=mint(0);
			return ret;
		}
		poly exp(const poly &a){
			uint l=a.size();if(l==1){poly ret(1);ret[0]=mint(1);return ret;}
			poly g0=a;g0.resize((l+1)>>1);g0=exp(g0);poly g1(g0);g1.resize(l);g1=ln(g1);
			for(uint i=0;i<l;++i) g1[i]=a[i]-g1[i];g1[0]+=mint(1);g1=mul(g1,g0);g1.resize(l);
			return g1;
		}
	};
	class fast_binomial_2_128
	{
	};
	class fast_binomial_2_64
	{
	};
	class fast_binomial_2_32
	{
	};
}
namespace tools
{
	#ifdef _OPENMP
		#include <omp.h>
	#else
		#define omp_get_thread_num()  0
		#define omp_get_num_threads() 1
	#endif
	/*
	* Code from https://stackoverflow.com/questions/28050669/can-i-report-progress-for-openmp-tasks
	*/
	class timer{
	private:
		typedef std::chrono::high_resolution_clock clock;
		typedef std::chrono::duration<double, std::ratio<1> > second;
		std::chrono::time_point<clock> start_time;
		double accumulated_time;
		bool running;
	public:
		timer(){
			accumulated_time = 0;
			running          = false;
		}
		void start(){
			if(running) throw std::runtime_error("Timer was already started!");
			running    = true;
			start_time = clock::now();
		}
		double stop(){
			if(!running) throw std::runtime_error("Timer was already stopped!");
			accumulated_time += lap();
			running           = false;

			return accumulated_time;
		}
		double accumulated(){
			if(running) throw std::runtime_error("Timer is still running!");
			return accumulated_time;
		}
		double lap(){
			if(!running) throw std::runtime_error("Timer was not started!");
			return std::chrono::duration_cast<second> (clock::now() - start_time).count();
		}
		void reset(){
			accumulated_time = 0;
			running          = false;
		}
		bool get_state(){
			return running;
		}
	};
	class progress_bar{
	private:
		uint32_t total_work;
		uint32_t next_update;
		uint32_t call_diff;
		uint32_t work_done;
		uint16_t old_percent;
		timer    _timer;
		void clear_console_line() const {
			std::cerr<<"\r\033[2K"<<std::flush;
		}
	public:
		void start(uint32_t total_work){
			_timer = timer();
			_timer.start();
			this->total_work = total_work;
			next_update      = 0;
			call_diff        = total_work/200;
			old_percent      = 0;
			work_done        = 0;
			clear_console_line();
		}
		void update(uint32_t work_done0,bool is_dynamic=true){
			#ifdef NO_PROGRESS
			  return;
			#endif
			if(omp_get_thread_num()!=0) return;
			work_done = work_done0;
			if(work_done<next_update)return;
			next_update += call_diff;
			uint16_t percent;
			#ifdef __INTEL_COMPILER 
			percent = (uint8_t)((uint64_t)work_done*omp_get_num_threads()*100/total_work);
			#else
			if(is_dynamic) percent = (uint8_t)((uint64_t)work_done*100/total_work);
			else percent = (uint8_t)((uint64_t)work_done*omp_get_num_threads()*100/total_work);
			#endif
			if(percent>100) percent=100;
			if(percent==old_percent) return;
			old_percent=percent;
			std::cerr<<"\r\033[2K["
					 <<std::string(percent/2, '=')<<std::string(50-percent/2, ' ')
					 <<"] ("
					 <<percent<<"% - "
					 <<std::fixed<<std::setprecision(1)<<_timer.lap()/percent*(100-percent)
					 <<"s - "
					 <<omp_get_num_threads()<< " threads)"<<std::flush;
		}
		progress_bar& operator++(){
			if(omp_get_thread_num()!=0) return *this;
			work_done++;
			update(work_done);
			return *this;
		}
		double stop(){
			clear_console_line();
			_timer.stop();
			return _timer.accumulated();
		}
		double time_it_took(){
			return _timer.accumulated();
		}
		uint32_t cells_processed() const {
			return work_done;
		}
		~progress_bar(){
			if(_timer.get_state()) this->stop();
		}
	};
}
#endif