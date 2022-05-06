#include <pe.hpp>

#if defined(__AVX__) && defined(__AVX2__)
#pragma message "AVX & AVX2 acceleration enabled."
#endif

#if defined(_OPENMP)
#pragma message "Openmp acceleration enabled."
#include <omp.h>
#else
	#define omp_get_thread_num()  0
	#define omp_get_num_threads() 1
#endif

namespace math
{
	namespace modulo{
		namespace mod_int{
			ui global_mod_mi=default_mod;
			fast_mod_32 global_fast_mod(default_mod);
			montgomery_mi_lib mi::mlib(default_mod);
			void set_mod_mi(ui p){
				mi::mlib=montgomery_mi_lib(p);
				global_fast_mod=fast_mod_32(p);
				global_mod_mi=p;
			}
			void set_mod_for_all_threads_mi(ui p){
				#if defined(_OPENMP)
				#pragma omp parallel
				{
				#endif
					mi::mlib=montgomery_mi_lib(p);
					global_fast_mod=fast_mod_32(p);
					global_mod_mi=p;
				#if defined(_OPENMP)
				}
				#endif
			}
			ull global_mod_mli=default_mod;
			montgomery_mli_lib mli::mlib(default_mod);
			void set_mod_mli(ull p){
				mli::mlib=montgomery_mli_lib(p);
				global_mod_mli=p;
			}
			void set_mod_for_all_threads_mli(ull p){
				#if defined(_OPENMP)
				#pragma omp parallel
				{
				#endif
					mli::mlib=montgomery_mli_lib(p);
					global_mod_mli=p;
				#if defined(_OPENMP)
				}
				#endif
			}
			#if defined(__AVX__) && defined(__AVX2__)
			ui global_mod_mai=default_mod;
			montgomery_mm256_lib mai::mlib(default_mod);
			void set_mod_mai(ui p){
				mai::mlib=montgomery_mm256_lib(p);
				global_mod_mai=p;
			}
			void set_mod_for_all_threads_mai(ui p){
				#if defined(_OPENMP)
				#pragma omp parallel
				{
				#endif
					mai::mlib=montgomery_mm256_lib(p);
					global_mod_mai=p;
				#if defined(_OPENMP)
				}
				#endif
			}
			#endif
		}
	}
	void power_series_ring::polynomial_kernel::polynomial_kernel_ntt::release(){
		ws0.reset();ws1.reset();
		_inv.reset();num.reset();
		fn=fb=mx=0;
		for(ui i=0;i<5;++i) tt[i].reset();
	}
	void power_series_ring::polynomial_kernel::polynomial_kernel_ntt::init(ui max_conv_size,ui P0,ui G0){
		ui pre_mi_mod=global_mod_mi;
		if(pre_mi_mod!=P0) set_mod_mi(P0);
		release();P=P0,G=G0;mx=max_conv_size;
		fn=1;fb=0;while(fn<(max_conv_size<<1)) fn<<=1,++fb;
		_inv=create_aligned_array<mi,32>(fn+32);ws0 =create_aligned_array<mi,32>(fn+32);
		ws1 =create_aligned_array<mi,32>(fn+32);num =create_aligned_array<mi,32>(fn+32);
		for(ui i=0;i<5;++i)	tt[i] =create_aligned_array<mi,32>(fn+32);
		_inv[0]=mi(1);for(ui i=2;i<=fn+32;++i) _inv[i-1]=(-mi(P/i))*_inv[(P%i)-1];
		for(ui i=1;i<=fn+32;++i) num[i-1]=mi(i);
		mi j0=basic::fast_pow(mi(G),(P-1)/fn),j1=basic::fast_pow(basic::fast_pow(mi(G),(P-2)),(P-1)/fn);
		for(ui mid=(fn>>1);mid>=1;mid>>=1,j0*=j0,j1*=j1){
			mi w0(1),w1(1);
			for(ui i=0;i<mid;++i,w0*=j0,w1*=j1) ws0[i+mid]=w0,ws1[i+mid]=w1;
		}
		if(pre_mi_mod!=P0) set_mod_mi(pre_mi_mod);
	}
	power_series_ring::polynomial_kernel::polynomial_kernel_ntt::polynomial_kernel_ntt(const polynomial_kernel_ntt &d){
		fn=d.fn,fb=d.fb;P=d.P,G=d.G;mx=d.mx;
		if(d.mx){
			_inv=create_aligned_array<mi,32>(fn+32);ws0 =create_aligned_array<mi,32>(fn+32);
			ws1 =create_aligned_array<mi,32>(fn+32);num =create_aligned_array<mi,32>(fn+32);
			for(ui i=0;i<5;++i)	tt[i] =create_aligned_array<mi,32>(fn+32);
			std::memcpy(ws0.get(), d.ws0.get(), sizeof(mi)*(fn+32));
			std::memcpy(ws1.get(), d.ws1.get(), sizeof(mi)*(fn+32));
			std::memcpy(_inv.get(),d._inv.get(),sizeof(mi)*(fn+32));
			std::memcpy(num.get(), d.num.get(), sizeof(mi)*(fn+32));
		}
	}
	power_series_ring::polynomial_kernel::polynomial_kernel_ntt::polynomial_kernel_ntt(){fn=fb=mx=0;}
	power_series_ring::polynomial_kernel::polynomial_kernel_ntt::polynomial_kernel_ntt(ui max_conv_size,ui P0,ui G0){init(max_conv_size,P0,G0);}
	void power_series_ring::polynomial_kernel::polynomial_kernel_ntt::dif(mi* restrict p,ui n){
		#if defined(__AVX__) && defined(__AVX2__)
		ui len=(1<<n);
		mi* restrict ws=ws0.get();
		if(len<8){
			mi t1,t2;
			for(ui l=len;l>=2;l>>=1) for(ui j=0,mid=(l>>1);j<len;j+=l){
				mi restrict *p1=p+j,*p2=p+j+mid,*ww=ws+mid;
				for(ui i=0;i<mid;++i,++p1,++p2,++ww) t1=*p1,t2=*p2,*p1=t1+t2,*p2=(t1-t2)*(*ww);
			}
		}else{
			__m256i* pp=(__m256i*)p,x,y,*p1,*p2,*ww;
			__m256i msk,val;
			for(ui l=len;l>8;l>>=1){
				ui mid=(l>>1);
				for(ui j=0;j<len;j+=l){
					p1=(__m256i*)(p+j),p2=(__m256i*)(p+j+mid),ww=(__m256i*)(ws+mid);
					for(ui i=0;i<mid;i+=8,++p1,++p2,++ww){
						x=*p1,y=*p2;
						*p1=mai::mlib.add(x,y);
						*p2=mai::mlib.mul(mai::mlib.sub(x,y),*ww);
					}
				}
			}
			val=_mm256_setr_epi32(ws[4].get_val(),ws[4].get_val(),ws[4].get_val(),ws[4].get_val(),
								  ws[4].get_val(),ws[5].get_val(),ws[6].get_val(),ws[7].get_val());
			msk=_mm256_setr_epi32(0,0,0,0,P*2,P*2,P*2,P*2);
			pp=(__m256i*)p;
			for(ui j=0;j<len;j+=8,++pp){
				x=_mm256_permute4x64_epi64(*pp,0x4E);
				y=_mm256_add_epi32(_mm256_sign_epi32(*pp,_mm256_setr_epi32(1,1,1,1,-1,-1,-1,-1)),msk);
				*pp=mai::mlib.mul(mai::mlib.redd(_mm256_add_epi32(x,y)),val);
			}
			val=_mm256_setr_epi32(ws[2].get_val(),ws[2].get_val(),ws[2].get_val(),ws[3].get_val(),
								  ws[2].get_val(),ws[2].get_val(),ws[2].get_val(),ws[3].get_val());
			msk=_mm256_setr_epi32(0,0,P*2,P*2,0,0,P*2,P*2);
			pp=(__m256i*)p;
			for(ui j=0;j<len;j+=8,++pp){
				x=_mm256_shuffle_epi32(*pp,0x4E);
				y=_mm256_add_epi32(_mm256_sign_epi32(*pp,_mm256_setr_epi32(1,1,-1,-1,1,1,-1,-1)),msk);
				*pp=mai::mlib.mul(mai::mlib.redd(_mm256_add_epi32(x,y)),val);
			}
			msk=_mm256_setr_epi32(0,P*2,0,P*2,0,P*2,0,P*2);
			pp=(__m256i*)p;
			for(ui j=0;j<len;j+=8,++pp){
				x=_mm256_shuffle_epi32(*pp,0xB1);
				y=_mm256_add_epi32(_mm256_sign_epi32(*pp,_mm256_setr_epi32(1,-1,1,-1,1,-1,1,-1)),msk);
				*pp=mai::mlib.redd(_mm256_add_epi32(x,y));
			}
		}
		#else
		ui len=(1<<n);
		mi t1,t2;
		mi* restrict ws=ws0.get();
		for(ui l=len;l>=2;l>>=1) for(ui j=0,mid=(l>>1);j<len;j+=l){
			mi restrict *p1=p+j,*p2=p+j+mid,*ww=ws+mid;
			for(ui i=0;i<mid;++i,++p1,++p2,++ww) t1=*p1,t2=*p2,*p1=t1+t2,*p2=(t1-t2)*(*ww);
		}
		#endif
	}
	void power_series_ring::polynomial_kernel::polynomial_kernel_ntt::dit(mi* restrict p,ui n){
		#if defined(__AVX__) && defined(__AVX2__)
		ui len=(1<<n);
		mi* restrict ws=ws1.get();
		if(len<8){
			mi t1,t2;
			for(ui l=2;l<=len;l<<=1) for(ui j=0,mid=(l>>1);j<len;j+=l){
				mi restrict *p1=p+j,*p2=p+j+mid,*ww=ws+mid;
				for(ui i=0;i<mid;++i,++p1,++p2,++ww) t1=*p1,t2=(*p2)*(*ww),*p1=t1+t2,*p2=t1-t2;
			}
			mi co=_inv[len-1];mi* restrict p1=p;
			for(ui i=0;i<len;++i,++p1) (*p1)*=co;
		}else{
			__m256i* pp=(__m256i*)p,x,y,*p1,*p2,*ww;
			__m256i msk,val;
			msk=_mm256_setr_epi32(0,P*2,0,P*2,0,P*2,0,P*2);
			pp=(__m256i*)p;
			for(ui j=0;j<len;j+=8,++pp){
				x=_mm256_shuffle_epi32(*pp,0xB1);
				y=_mm256_add_epi32(_mm256_sign_epi32(*pp,_mm256_setr_epi32(1,-1,1,-1,1,-1,1,-1)),msk);
				*pp=mai::mlib.redd(_mm256_add_epi32(x,y));
			}
			val=_mm256_setr_epi32(ws[2].get_val(),ws[3].get_val(),(-ws[2]).get_val(),(-ws[3]).get_val(),
								  ws[2].get_val(),ws[3].get_val(),(-ws[2]).get_val(),(-ws[3]).get_val());
			pp=(__m256i*)p;
			for(ui j=0;j<len;j+=8,++pp){
				x=_mm256_shuffle_epi32(*pp,0x44);
				y=_mm256_shuffle_epi32(*pp,0xEE);
				*pp=mai::mlib.add(x,mai::mlib.mul(y,val));
			}
			val=_mm256_setr_epi32(  ws[4].get_val(),   ws[5].get_val(),   ws[6].get_val(),   ws[7].get_val(),
								  (-ws[4]).get_val(),(-ws[5]).get_val(),(-ws[6]).get_val(),(-ws[7]).get_val());
			pp=(__m256i*)p;
			for(ui j=0;j<len;j+=8,++pp){
				x=_mm256_permute4x64_epi64(*pp,0x44);
				y=_mm256_permute4x64_epi64(*pp,0xEE);
				*pp=mai::mlib.add(x,mai::mlib.mul(y,val));
			}
			for(ui l=16;l<=len;l<<=1){
				ui mid=(l>>1);
				for(ui j=0;j<len;j+=l){
					p1=(__m256i*)(p+j),p2=(__m256i*)(p+j+mid),ww=(__m256i*)(ws+mid);
					for(ui i=0;i<mid;i+=8,++p1,++p2,++ww){
						x=*p1,y=mai::mlib.mul(*p2,*ww);
						*p1=mai::mlib.add(x,y);
						*p2=mai::mlib.sub(x,y);
					}
				}
			}
			__m256i co=_mm256_set1_epi32(_inv[len-1].get_val());
			pp=(__m256i*)p;
			for(ui i=0;i<len;i+=8,++pp) (*pp)=mai::mlib.mul(*pp,co);
		}
		#else
		ui len=(1<<n);
		mi t1,t2;
		mi* restrict ws=ws1.get();
		for(ui l=2;l<=len;l<<=1) for(ui j=0,mid=(l>>1);j<len;j+=l){
			mi restrict *p1=p+j,*p2=p+j+mid,*ww=ws+mid;
			for(ui i=0;i<mid;++i,++p1,++p2,++ww) t1=*p1,t2=(*p2)*(*ww),*p1=t1+t2,*p2=t1-t2;
		}
		mi co=_inv[len-1];mi* restrict p1=p;
		for(ui i=0;i<len;++i,++p1) (*p1)*=co;
		#endif
	}
	void power_series_ring::polynomial_kernel::polynomial_kernel_ntt::internal_mul(mi* restrict src1,mi* restrict src2,mi* restrict dst,ui m)
	{
		dif(src1,m);
		dif(src2,m);
		#if defined (__AVX__) && defined(__AVX2__)
		if((1<<m)<8){
			for(ui i=0;i<(1<<m);++i) dst[i]=src1[i]*src2[i];
		}
		else{
			__m256i restrict *p1=(__m256i*)src1, *p2=(__m256i*)src2, *p3=(__m256i*)dst;
			for(ui i=0;i<(1<<m);i+=8,++p1,++p2,++p3) *p3=mai::mlib.mul(*p1,*p2);
		}
		#else
		for(ui i=0;i<(1<<m);++i) dst[i]=src1[i]*src2[i];
		#endif
		dit(dst,m);
	}
	power_series_ring::poly power_series_ring::polynomial_kernel::polynomial_kernel_ntt::mul(const power_series_ring::poly &a,const power_series_ring::poly &b){
		ui la=a.size(),lb=b.size();if((!la) && (!lb)) return poly();
		if(la>mx || lb>mx) throw std::runtime_error("Convolution size out of range!");
		ui pre_mi_mod=global_mod_mi;
		if(pre_mi_mod!=P) set_mod_mi(P);
		#if defined(__AVX__) && defined(__AVX2__)
		ui pre_mai_mod=global_mod_mai;
		if(pre_mai_mod!=P) set_mod_mai(P);
		#endif
		ui m=0;if(la+lb>2) m=32-__builtin_clz(la+lb-2);
		std::memcpy(tt[0].get(),&a[0],sizeof(mi)*la);std::memset(tt[0].get()+la,0,sizeof(mi)*((1<<m)-la));
		std::memcpy(tt[1].get(),&b[0],sizeof(mi)*la);std::memset(tt[1].get()+lb,0,sizeof(mi)*((1<<m)-lb));
		internal_mul(tt[0].get(),tt[1].get(),tt[2].get(),m);
		poly ret(la+lb-1);
		std::memcpy(&ret[0],tt[2].get(),sizeof(mi)*(la+lb-1));
		if(pre_mi_mod!=P) set_mod_mi(pre_mi_mod);
		#if defined(__AVX__) && defined(__AVX2__)
		if(pre_mai_mod!=P) set_mod_mai(pre_mai_mod);
		#endif
		return ret;
	}
	void power_series_ring::polynomial_kernel::polynomial_kernel_ntt::internal_inv(mi* restrict src,mi* restrict dst,mi* restrict tmp,ui len){
		if(len==1){dst[0]=basic::fast_pow(src[0],P-2);return;}
		internal_inv(src,dst,tmp,len>>1);
		std::memcpy(tmp,src,sizeof(mi)*len);std::memset(dst+(len>>1),0,sizeof(mi)*(len>>1)*3);
		std::memset(tmp+len,0,sizeof(mi)*len);
		dif(tmp,__builtin_ctz(len<<1));dif(dst,__builtin_ctz(len<<1));
		#if defined(__AVX__) && defined(__AVX2__)
		if(len<4){
			mi m2(2);
			for(ui i=0;i<(len<<1);++i) dst[i]*=(m2-tmp[i]*dst[i]);
		}
		else{
			__m256i restrict *p1=(__m256i*)dst,*p2=(__m256i*)tmp;
			__m256i m2=_mm256_set1_epi32(mi(2).get_val());
			for(ui i=0;i<(len<<1);i+=8,++p1,++p2) *p1=mai::mlib.mul(*p1,mai::mlib.sub(m2,mai::mlib.mul(*p1,*p2)));
		}
		#else
		mi m2(2);
		for(ui i=0;i<(len<<1);++i) dst[i]*=(m2-tmp[i]*dst[i]);
		#endif
		dit(dst,__builtin_ctz(len<<1));
	}
	power_series_ring::poly power_series_ring::polynomial_kernel::polynomial_kernel_ntt::inv(const power_series_ring::poly &src){
		ui la=src.size();if(!la) throw std::runtime_error("Inversion calculation of empty polynomial!");
		if(la>mx) throw std::runtime_error("Convolution size out of range!");
		ui pre_mi_mod=global_mod_mi;
		if(pre_mi_mod!=P) set_mod_mi(P);
		#if defined(__AVX__) && defined(__AVX2__)
		ui pre_mai_mod=global_mod_mai;
		if(pre_mai_mod!=P) set_mod_mai(P);
		#endif
		if(!src[0].real_val()){
			if(pre_mi_mod!=P) set_mod_mi(pre_mi_mod);
			#if defined(__AVX__) && defined(__AVX2__)
			if(pre_mai_mod!=P) set_mod_mai(pre_mai_mod);
			#endif
			throw std::runtime_error("Inversion calculation of polynomial which has constant not equal to 1!");
		}
		ui m=0;if(la>1) m=32- __builtin_clz(la-1);
		std::memcpy(tt[0].get(),&src[0],sizeof(mi)*la);std::memset(tt[0].get()+la,0,sizeof(mi)*((1<<m)-la));
		internal_inv(tt[0].get(),tt[1].get(),tt[2].get(),(1<<m));
		poly ret(la);
		std::memcpy(&ret[0],tt[1].get(),sizeof(mi)*la);
		if(pre_mi_mod!=P) set_mod_mi(pre_mi_mod);
		#if defined(__AVX__) && defined(__AVX2__)
		if(pre_mai_mod!=P) set_mod_mai(pre_mai_mod);
		#endif
		return ret;
	}
	power_series_ring::polynomial_kernel::polynomial_kernel_ntt::~polynomial_kernel_ntt(){release();}
	void power_series_ring::polynomial_kernel::polynomial_kernel_ntt::internal_ln(mi* restrict src,mi* restrict dst,mi* restrict tmp1,mi* restrict tmp2,ui len){
		#if defined(__AVX__) && defined(__AVX2__)
		ui pos=1;__m256i restrict *pp=(__m256i*)tmp1,*iv=(__m256i*)num.get();mi restrict *p1=src+1;
		for(;pos+8<=len;pos+=8,p1+=8,++pp,++iv) *pp=mai::mlib.mul(_mm256_loadu_si256((__m256i*)p1),*iv);
		for(;pos<len;++pos) tmp1[pos-1]=src[pos]*num[pos-1];tmp1[len-1]=mi(0);
		#else
		mi restrict *p1=src+1,*p2=tmp1,*p3=num.get();
		for(ui i=1;i<len;++i,++p1,++p2,++p3) *p2=(*p1)*(*p3);
		tmp1[len-1]=mi(0);
		#endif
		internal_inv(src,dst,tmp2,len);
		std::memset(dst+len,0,sizeof(mi)*len);std::memset(tmp1+len,0,sizeof(mi)*len);
		internal_mul(tmp1,dst,tmp2,__builtin_ctz(len<<1));
		#if defined(__AVX__) && defined(__AVX2__)
		ui ps=1;__m256i restrict *pp0=(__m256i*)tmp2,*iv0=(__m256i*)_inv.get();mi restrict *p10=dst+1;
		for(;ps+8<=len;ps+=8,p10+=8,++pp0,++iv0) _mm256_storeu_si256((__m256i*)p10,mai::mlib.mul(*pp0,*iv0));
		dst[0]=mi(0);
		for(;ps<len;++ps) dst[ps]=_inv[ps-1]*tmp2[ps-1];
		#else
		dst[0]=mi(0);
		mi restrict *p10=dst+1,*p20=tmp2,*p30=_inv.get();
		for(ui i=1;i<len;++i,++p10,++p20,++p30) *p10=(*p20)*(*p30);
		#endif
	}
	power_series_ring::poly power_series_ring::polynomial_kernel::polynomial_kernel_ntt::ln(const poly &src){
		ui la=src.size();if(!la) throw std::runtime_error("Ln calculation of empty polynomial!");
		if(la>mx) throw std::runtime_error("Convolution size out of range!");
		ui pre_mi_mod=global_mod_mi;
		if(pre_mi_mod!=P) set_mod_mi(P);
		#if defined(__AVX__) && defined(__AVX2__)
		ui pre_mai_mod=global_mod_mai;
		if(pre_mai_mod!=P) set_mod_mai(P);
		#endif
		if(src[0].real_val()!=1){
			if(pre_mi_mod!=P) set_mod_mi(pre_mi_mod);
			#if defined(__AVX__) && defined(__AVX2__)
			if(pre_mai_mod!=P) set_mod_mai(pre_mai_mod);
			#endif
			throw std::runtime_error("Ln calculation of polynomial which has constant not equal to 1!");
		}
		ui m=0;if(la>1) m=32- __builtin_clz(la-1);
		std::memcpy(tt[0].get(),&src[0],sizeof(mi)*la);std::memset(tt[0].get()+la,0,sizeof(mi)*((1<<m)-la));
		internal_ln(tt[0].get(),tt[1].get(),tt[2].get(),tt[3].get(),(1<<m));
		poly ret(la);
		std::memcpy(&ret[0],tt[1].get(),sizeof(mi)*la);
		if(pre_mi_mod!=P) set_mod_mi(pre_mi_mod);
		#if defined(__AVX__) && defined(__AVX2__)
		if(pre_mai_mod!=P) set_mod_mai(pre_mai_mod);
		#endif
		return ret;
	}
	std::pair<long long,long long> power_series_ring::polynomial_kernel::polynomial_kernel_ntt::test(ui T){
		ui pre_mi_mod=global_mod_mi;
		if(pre_mi_mod!=P) set_mod_mi(P);
		#if defined(__AVX__) && defined(__AVX2__)
		ui pre_mai_mod=global_mod_mai;
		if(pre_mai_mod!=P) set_mod_mai(P);
		#endif
		std::mt19937 rnd(default_mod);std::uniform_int_distribution<ui> rng{0,default_mod-1};
		ui len=(fn>>1);
		for(ui i=0;i<len;++i) tt[0][i]=tt[1][i]=mi(rng(rnd));
		auto dif_start=std::chrono::system_clock::now();
		for(ui i=0;i<T;++i) dif(tt[0].get(),__builtin_ctz(len));
		auto dif_end=std::chrono::system_clock::now();
		auto dit_start=std::chrono::system_clock::now();
		for(ui i=0;i<T;++i) dit(tt[0].get(),__builtin_ctz(len));
		auto dit_end=std::chrono::system_clock::now();
		for(ui i=0;i<len;++i) assert(tt[0][i].real_val()==tt[1][i].real_val());
		auto dif_duration=std::chrono::duration_cast<std::chrono::microseconds>(dif_end-dif_start),
			 dit_duration=std::chrono::duration_cast<std::chrono::microseconds>(dit_end-dit_start);
		if(pre_mi_mod!=P) set_mod_mi(pre_mi_mod);
		#if defined(__AVX__) && defined(__AVX2__)
		if(pre_mai_mod!=P) set_mod_mai(pre_mai_mod);
		#endif
		return std::make_pair(dif_duration.count(),dit_duration.count());
	}
}
namespace tools
{
	/*
	* Code from https://stackoverflow.com/questions/28050669/can-i-report-progress-for-openmp-tasks
	*/
	timer::timer(){
		accumulated_time = 0;
		running          = false;
	}
	void timer::start(){
		if(running) throw std::runtime_error("Timer was already started!");
		running    = true;
		start_time = clock::now();
	}
	double timer::stop(){
		if(!running) throw std::runtime_error("Timer was already stopped!");
		accumulated_time += lap();
		running           = false;

		return accumulated_time;
	}
	double timer::accumulated(){
		if(running) throw std::runtime_error("Timer is still running!");
		return accumulated_time;
	}
	double timer::lap(){
		if(!running) throw std::runtime_error("Timer was not started!");
		return std::chrono::duration_cast<second> (clock::now() - start_time).count();
	}
	void timer::reset(){
		accumulated_time = 0;
		running          = false;
	}
	bool timer::get_state(){
		return running;
	}
	void progress_bar::clear_console_line() const {
		std::cerr<<"\r\033[2K"<<std::flush;
	}
	void progress_bar::start(uint32_t total_work){
		_timer = timer();
		_timer.start();
		this->total_work = total_work;
		next_update      = 0;
		call_diff        = total_work/200;
		old_percent      = 0;
		work_done        = 0;
		clear_console_line();
	}
	void progress_bar::update(uint32_t work_done0,bool is_dynamic){
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
	progress_bar& progress_bar::operator++(){
		if(omp_get_thread_num()!=0) return *this;
		work_done++;
		update(work_done);
		return *this;
	}
	double progress_bar::stop(){
		clear_console_line();
		_timer.stop();
		return _timer.accumulated();
	}
	double progress_bar::time_it_took(){
		return _timer.accumulated();
	}
	uint32_t progress_bar::cells_processed() const {
		return work_done;
	}
	progress_bar::~progress_bar(){
		if(_timer.get_state()) this->stop();
	}
}