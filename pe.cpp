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
	int global_mod=default_mod;
	FastMod global_fast_mod=FastMod(default_mod);
	mint fast_pow(mint a,ull b){mint ans=mint(1),off=a;while(b){if(b&1) ans*=off;off*=off;b>>=1;}return ans;}
	montgomery_int_lib montgomery_int::mlib=montgomery_int_lib(default_mod);
	void set_mint_mod(uint32_t p){
		mint::mlib=montgomery_int_lib(p);
	}
	#if defined(__AVX__) && defined(__AVX2__)
	montgomery_mm256_lib montgomery_mm256_int::mlib=montgomery_mm256_lib(default_mod);
	void set_m256int_mod(uint32_t p){
		m256int::mlib=montgomery_mm256_lib(p);
	}
	#endif
	void set_mod_for_all_threads(uint32_t p){
		#if defined(_OPENMP)
		#pragma omp parallel
		{
		#endif
			global_mod=p;
			global_fast_mod=FastMod(p);
			set_mint_mod(p);
			#if defined(__AVX__) && defined(__AVX2__)
			set_m256int_mod(p);
			#endif
		#if defined(_OPENMP)
		}
		#endif
	}
	void set_mod(uint32_t p){
		global_mod=p;
		global_fast_mod=FastMod(p);
		set_mint_mod(p);
		#if defined(__AVX__) && defined(__AVX2__)
		set_m256int_mod(p);
		#endif
	}
	uint fast_binomial_2_32::fast_pow(uint a,uint b){uint ans=1,off=a;while(b){if(b&1) ans=ans*off;off=off*off;b>>=1;}return ans;}
	uint fast_binomial_2_32::calc_bju(uint j,uint u){
		ull cur=0,u2=1ull*u*u;
		for(uint i=15;~i;--i) cur=cur*u2+coefs[j][i];cur*=u;
		return fast_pow(bases[j],(cur>>32));
	}
	uint fast_binomial_2_32::odd_factorial(uint k){
		uint ans=1;uint k0=(k>>1);if(k&1) ++k0;
		for(int j=0;j<16;++j) ans*=calc_bju(j,k0);
		if((ans&2)!=((k0+1)&2)) ans=-ans;
		return ans;
	}
	uint fast_binomial_2_32::factorial_odd(uint k){
		uint ans=1;
		while(k){ans*=odd_factorial(k);k>>=1;}
		return ans;
	}
	uint fast_binomial_2_32::factorial(uint k){
		uint cnt2=0;
		while(k){cnt2+=(k>>1);k>>=1;}
		if(cnt2>=32) return 0;
		return factorial_odd(k)<<cnt2;
	}
	uint fast_binomial_2_32::binomial(uint upper,uint lower){
		if(upper<lower) return 0;uint cnt2=__builtin_popcount(lower)+__builtin_popcount(upper-lower)-__builtin_popcount(upper);
		if(cnt2>=32) return 0;
		uint c=fast_pow(factorial_odd(lower)*factorial_odd(upper-lower),(1u<<31)-1)*factorial_odd(upper);
		return c<<cnt2;
	}
	ull fast_binomial_2_64::fast_pow(ull a,ull b){ull ans=1,off=a;while(b){if(b&1) ans=ans*off;off=off*off;b>>=1;}return ans;}
	ull fast_binomial_2_64::calc_bju(ull j,ull u){
		__uint128_t cur=0,u2=__uint128_t(u)*u;
		for(uint i=31;~i;--i) cur=cur*u2+coefs[j][i];cur*=u;
		return fast_pow(bases[j],(cur>>64));
	}
	ull fast_binomial_2_64::odd_factorial(ull k){
		ull ans=1;ull k0=(k>>1);if(k&1) ++k0;
		for(int j=0;j<32;++j) ans*=calc_bju(j,k0);
		if((ans&2)!=((k0+1)&2)) ans=-ans;
		return ans;
	}
	ull fast_binomial_2_64::factorial_odd(ull k){
		ull ans=1;
		while(k){ans*=odd_factorial(k);k>>=1;}
		return ans;
	}
	ull fast_binomial_2_64::factorial(ull k){
		uint cnt2=0;
		while(k){cnt2+=(k>>1);k>>=1;}
		if(cnt2>=64) return 0;
		return factorial_odd(k)<<cnt2;
	}
	ull fast_binomial_2_64::binomial(ull upper,ull lower){
		if(upper<lower) return 0;uint cnt2=__builtin_popcountll(lower)+__builtin_popcountll(upper-lower)-__builtin_popcountll(upper);
		if(cnt2>=64) return 0;
		ull c=fast_pow(factorial_odd(lower)*factorial_odd(upper-lower),(1ull<<63)-1)*factorial_odd(upper);
		return c<<cnt2;
	}
	void polynomial_ntt::release(){
		ws0.reset();ws1.reset();fac.reset();ifac.reset();_inv.reset();tt.reset();
		rev.reset();lgg.reset();
	}
	void polynomial_ntt::init(uint max_conv_size,uint P0,uint G0){
		release();P=P0,G=G0;
		fn=1;fb=0;while(fn<(max_conv_size<<1)) fn<<=1,++fb;
		_inv=create_aligned_array<mint,32>(fn+32);ws0 =create_aligned_array<mint,32>(fn+32);
		ws1 =create_aligned_array<mint,32>(fn+32);fac =create_aligned_array<mint,32>(fn+32);
		ifac=create_aligned_array<mint,32>(fn+32);tt  =create_aligned_array<mint,32>(fn+32);
		rev =create_aligned_array<uint,32>(fn+32);lgg =create_aligned_array<uint,32>(fn+32);
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
	polynomial_ntt::polynomial_ntt(const polynomial_ntt &d){
		fn=d.fn,fb=d.fb;P=d.P,G=d.G;
		if(d.rev){
			_inv=create_aligned_array<mint,32>(fn+32);ws0 =create_aligned_array<mint,32>(fn+32);
			ws1 =create_aligned_array<mint,32>(fn+32);fac =create_aligned_array<mint,32>(fn+32);
			ifac=create_aligned_array<mint,32>(fn+32);tt  =create_aligned_array<mint,32>(fn+32);
			rev =create_aligned_array<uint,32>(fn+32);lgg =create_aligned_array<uint,32>(fn+32);
			std::memcpy(ws0.get(), d.ws0.get(), sizeof(mint)*(fn+32));
			std::memcpy(ws1.get(), d.ws1.get(), sizeof(mint)*(fn+32));
			std::memcpy(fac.get(), d.fac.get(), sizeof(mint)*(fn+32));
			std::memcpy(ifac.get(),d.ifac.get(),sizeof(mint)*(fn+32));
			std::memcpy(_inv.get(),d._inv.get(),sizeof(mint)*(fn+32));
			std::memcpy(tt.get(),  d.tt.get(),  sizeof(mint)*(fn+32));
			std::memcpy(rev.get(), d.rev.get(), sizeof(uint)*(fn+32));
			std::memcpy(lgg.get(), d.lgg.get(), sizeof(uint)*(fn+32));
		}
	}
	polynomial_ntt::polynomial_ntt(){}
	// TODO: add avx support
	// #if defined(__AVX__) && defined(__AVX2__)
	// #else
	void polynomial_ntt::NTT(poly &p,int V){
		uint bts=lgg[p.size()];if(p.size()!=(1<<bts)) p.resize((1<<bts));
		mint *w=(V==1)?ws0.get():ws1.get();uint len=(1<<bts);for(uint i=0;i<len;++i) tt[i]=p[rev[i]>>(fb-bts)];
		mint t1,t2;
		for(uint l=2;l<=len;l<<=1)
			for(uint j=0,mid=(l>>1);j<len;j+=l)
				for(uint i=0;i<mid;++i) t1=tt[j+i],t2=tt[j+i+mid]*w[mid+i],tt[j+i]=t1+t2,tt[j+i+mid]=t1-t2;
		if(V==1) for(uint i=0;i<len;++i) p[i]=tt[i];
		else{mint j=_inv[len];for(uint i=0;i<len;++i) p[i]=tt[i]*j;}
	}
	poly polynomial_ntt::mul(const poly &a,const poly &b){
		poly p1(a),p2(b);uint len=a.size()+b.size()-1,ff=(1<<lgg[len]);
		p1.resize(ff),p2.resize(ff);NTT(p1,1);NTT(p2,1);
		for(uint i=0;i<ff;++i) p1[i]*=p2[i];NTT(p1,-1);
		p1.resize(len);return p1;
	}
	poly polynomial_ntt::inv(const poly &a){
		uint l=a.size();if(l==1){poly ret(1);ret[0]=fast_pow(mint(a[0]),P-2);return ret;}
		poly g0=a;g0.resize((l+1)>>1);g0=inv(g0);
		poly p1(a);uint ff=(2<<lgg[l]);g0.resize(ff);p1.resize(ff);mint m2(2);
		NTT(p1,1);NTT(g0,1);for(uint i=0;i<ff;++i) g0[i]=g0[i]*(m2-g0[i]*p1[i]);
		NTT(g0,-1);g0.resize(l);return g0;
	}
	poly polynomial_ntt::ln(const poly &a){
		uint l=a.size();poly p1(l-1);
		for(uint i=1;i<l;++i) p1[i-1]=mint(i)*a[i];
		p1=mul(p1,inv(a));p1.resize(l-1);poly ret(l);
		for(uint i=1;i<l;++i) ret[i]=_inv[i]*p1[i-1];ret[0]=mint(0);
		return ret;
	}
	poly polynomial_ntt::exp(const poly &a){
		uint l=a.size();if(l==1){poly ret(1);ret[0]=mint(1);return ret;}
		poly g0=a;g0.resize((l+1)>>1);g0=exp(g0);poly g1(g0),g2(a);g1.resize(l);g1=ln(g1);
		uint ff=(2<<lgg[l]);g0.resize(ff);g1.resize(ff);g2.resize(ff);
		NTT(g0,1);NTT(g1,1);NTT(g2,1);mint m1(1);
		for(uint i=0;i<ff;++i) g0[i]=g0[i]*(m1-g1[i]+g2[i]);
		NTT(g0,-1);g0.resize(l);return g0;
	}
	// #endif
	polynomial_ntt::~polynomial_ntt(){release();}
	void polynomial::release(){
		pn1.release();
		pn2.release();
		pn3.release();
		_inv.reset();
	}
	polynomial::polynomial(const polynomial &d):pn1(d.pn1),pn2(d.pn2),pn3(d.pn3),F1(P1),F2(P2),F3(P3){
		if(d._inv){
			_inv=create_aligned_array<mint,32>(d.pn1.fn+32);P=d.P;F=FastMod(P);N3=1ull*P1*P2%P;
			memcpy(_inv.get(),d._inv.get(),sizeof(mint)*(d.pn1.fn+32));
		}
	}
	polynomial::polynomial():F1(P2),F2(P2),F3(P3){}
	polynomial::~polynomial(){release();}
	void polynomial::init(uint max_conv_size,uint P0){
		release();P=P0;F=FastMod(P);
		set_mod(P1);
		pn1.init(max_conv_size,P1,3);
		set_mod(P2);
		pn2.init(max_conv_size,P2,3);
		set_mod(P3);
		pn3.init(max_conv_size,P3,3);
		set_mod(P);
		_inv=create_aligned_array<mint,32>(pn1.fn+32);_inv[1]=mint(1);
		for(uint i=2;i<=pn1.fn;++i) _inv[i]=(-mint(P/i))*_inv[P%i];
	}
	poly polynomial::mul(const poly &a,const poly &b){
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
	poly polynomial::inv(const poly &a){
		uint l=a.size();if(l==1){poly ret(1);ret[0]=fast_pow(mint(a[0]),P-2);return ret;}
		poly g0=a;g0.resize((l+1)>>1);g0=inv(g0);poly g1=mul(a,g0);g1.resize(l);
		for(uint i=0;i<l;++i) g1[i]=-g1[i];g1[0]+=mint(2);g1=mul(g1,g0);g1.resize(l);
		return g1;
	}
	poly polynomial::ln(const poly &a){
		uint l=a.size();poly p1(l-1);
		for(uint i=1;i<l;++i) p1[i-1]=mint(i)*a[i];
		p1=mul(p1,inv(a));p1.resize(l-1);poly ret(l);
		for(uint i=1;i<l;++i) ret[i]=_inv[i]*p1[i-1];ret[0]=mint(0);
		return ret;
	}
	poly polynomial::exp(const poly &a){
		uint l=a.size();if(l==1){poly ret(1);ret[0]=mint(1);return ret;}
		poly g0=a;g0.resize((l+1)>>1);g0=exp(g0);poly g1(g0);g1.resize(l);g1=ln(g1);
		for(uint i=0;i<l;++i) g1[i]=a[i]-g1[i];g1[0]+=mint(1);g1=mul(g1,g0);g1.resize(l);
		return g1;
	}
	linear_modulo_preprocessing::linear_modulo_preprocessing(){}
	linear_modulo_preprocessing::~linear_modulo_preprocessing(){release();}
	void linear_modulo_preprocessing::release(){
		if(_inv){
			_inv.reset();fac.reset();ifac.reset();
		}
	}
	void linear_modulo_preprocessing::init(uint maxn,uint P0){
		release();rg=maxn;P=P0;
		fac=std::make_unique<mint[]>(rg+32);ifac=std::make_unique<mint[]>(rg+32);_inv=std::make_unique<mint[]>(rg+32);
		_inv[0]=0,_inv[1]=fac[0]=ifac[0]=1;
		for(uint i=2;i<rg+32;++i) _inv[i]=(-mint(P/i))*_inv[P%i];
		for(uint i=1;i<rg+32;++i) fac[i]=fac[i-1]*mint(i),ifac[i]=ifac[i-1]*_inv[i];
	}
	linear_modulo_preprocessing::linear_modulo_preprocessing(const linear_modulo_preprocessing &d){
		if(d._inv){
			rg=d.rg;P=d.P;
			fac=std::make_unique<mint[]>(rg+32);ifac=std::make_unique<mint[]>(rg+32);_inv=std::make_unique<mint[]>(rg+32);
			std::memcpy(fac.get(),d.fac.get(),sizeof(mint)*(rg+32));std::memcpy(ifac.get(),d.ifac.get(),sizeof(mint)*(rg+32));memcpy(_inv.get(),d._inv.get(),sizeof(mint)*(rg+32));
		}
	}
	mint linear_modulo_preprocessing::inverse(uint i) const {return _inv[i];}
	mint linear_modulo_preprocessing::inverse_factorial(uint i) const {return ifac[i];}
	mint linear_modulo_preprocessing::factorial(uint i) const {return fac[i];}
	mint linear_modulo_preprocessing::binomial(uint upper,uint lower) const {if(lower>upper) return mint();return fac[upper]*ifac[lower]*ifac[upper-lower];}
	void sieve::release(){
		mnf.reset();_mu.reset();phi.reset();mnfc.reset();pksum.reset();d.reset();ds.reset();
		premu.reset();preei.reset();pred.reset(),preds.reset();ps.clear();
	}
	sieve::~sieve(){release();}
	sieve::sieve(){}
	void sieve::init(uint maxn,int flg){
		release();
		flg|=(flg>>sieve_prefix_sum_offset);rg=maxn;pc=0;
		mnf=std::make_unique<uint[]>(maxn+1);vis=std::make_unique<bool[]>(maxn+1);
		if(flg&sieve_mu) _mu=std::make_unique<int[]>(maxn+1),_mu[0]=0,_mu[1]=1;
		if(flg&sieve_euler_phi) phi=std::make_unique<uint[]>(maxn+1),phi[0]=0,phi[1]=1;
		if(flg&sieve_divisors) mnfc=std::make_unique<uint[]>(maxn+1),d=std::make_unique<uint[]>(maxn+1),mnfc[0]=mnfc[1]=0,d[0]=0,d[1]=1;
		if(flg&sieve_divisors_sum) pksum=std::make_unique<uint[]>(maxn+1),ds=std::make_unique<uint[]>(maxn+1),pksum[0]=pksum[1]=1,ds[0]=0,ds[1]=1;
		if(flg&sieve_prefix_sum_mu) premu=std::make_unique<ll[]>(maxn+1),premu[0]=0;
		if(flg&sieve_prefix_sum_euler_phi) preei=std::make_unique<ll[]>(maxn+1),preei[0]=0;
		if(flg&sieve_prefix_sum_divisors) pred=std::make_unique<ll[]>(maxn+1),pred[0]=0;
		if(flg&sieve_prefix_sum_divisors_sum) preds=std::make_unique<ll[]>(maxn+1),preds[0]=0;
		std::memset(vis.get(),0,sizeof(bool)*(maxn+1));
		mnf[0]=0,mnf[1]=1;
		for(uint i=2;i<=maxn;++i){
			if(!vis[i]){
				ps.push_back(i);++pc;
				mnf[i]=pc;
				if(flg&sieve_mu) _mu[i]=-1;
				if(flg&sieve_euler_phi) phi[i]=i-1;
				if(flg&sieve_divisors) mnfc[i]=1,d[i]=2;
				if(flg&sieve_divisors_sum) pksum[i]=i+1,ds[i]=i+1;
			}
			for(uint j=0;j<pc && 1ull*ps[j]*i<=maxn;++j){
				uint t=ps[j]*i;vis[t]=true;mnf[t]=j+1;
				if((i%ps[j])==0){
					if(flg&sieve_mu) _mu[t]=0;
					if(flg&sieve_euler_phi) phi[t]=phi[i]*ps[j];
					if(flg&sieve_divisors) mnfc[t]=mnfc[i]+1,d[t]=d[i]/(mnfc[i]+1)*(mnfc[t]+1);
					if(flg&sieve_divisors_sum) pksum[t]=pksum[i]*ps[j]+1,ds[t]=ds[i]/pksum[i]*pksum[t];
					break;
				}
				else{
					if(flg&sieve_mu) _mu[t]=-_mu[i];
					if(flg&sieve_euler_phi) phi[t]=phi[i]*(ps[j]-1);
					if(flg&sieve_divisors) mnfc[t]=1,d[t]=d[i]*2;
					if(flg&sieve_divisors_sum) pksum[t]=ps[j]+1,ds[t]=ds[i]*(ps[j]+1);
				}
			}
		}
		if(flg>>sieve_prefix_sum_offset){
			for(uint i=1;i<=maxn;++i){
				if(flg&sieve_prefix_sum_mu) premu[i]=premu[i-1]+_mu[i];
				if(flg&sieve_prefix_sum_euler_phi) preei[i]=preei[i-1]+phi[i];
				if(flg&sieve_prefix_sum_divisors) pred[i]=pred[i-1]+d[i];
				if(flg&sieve_prefix_sum_divisors_sum) preds[i]=preds[i-1]+ds[i];
			}
		}
	}
	uint sieve::prime_count() const {return pc;}
	uint sieve::nth_prime(uint k) const {return ps[k-1];}
	std::vector<uint> sieve::all_primes() const {return ps;}
	bool sieve::is_prime(uint k) const {if(k<=1) return false;else return !vis[k];}
	uint sieve::min_prime_factor(uint k) const {return ps[mnf[k]-1];}
	uint sieve::prime_index(uint k) const {return mnf[k];}
	int sieve::mu(uint k) const {return _mu[k];}
	uint sieve::euler_phi(uint k) const {return phi[k];}
	uint sieve::divisors(uint k) const {return d[k];}
	uint sieve::divisors_sum(uint k) const {return ds[k];}
	ll sieve::prefix_sum_mu(uint k) const {return premu[k];}
	ll sieve::prefix_sum_euler_phi(uint k) const {return preei[k];}
	ll sieve::prefix_sum_divisors(uint k) const {return pred[k];}
	ll sieve::prefix_sum_divisors_sum(uint k) const {return preds[k];}
	sieve::sieve(const sieve &t){
		rg=t.rg,pc=t.pc,ps=t.ps;
		if(t.mnf){
			mnf=std::make_unique<uint[]>(rg+1);
			std::memcpy(mnf.get(),t.mnf.get(),sizeof(uint)*(rg+1));
		}
		if(t.vis){
			vis=std::make_unique<bool[]>(rg+1);
			std::memcpy(vis.get(),t.vis.get(),sizeof(bool)*(rg+1));
		}
		if(t._mu){
			_mu=std::make_unique<int[]>(rg+1);
			std::memcpy(_mu.get(),t._mu.get(),sizeof(int)*(rg+1));
		}
		if(t.phi){
			phi=std::make_unique<uint[]>(rg+1);
			std::memcpy(phi.get(),t.phi.get(),sizeof(uint)*(rg+1));
		}
		if(t.mnfc){
			mnfc=std::make_unique<uint[]>(rg+1);
			std::memcpy(mnfc.get(),t.mnfc.get(),sizeof(uint)*(rg+1));
		}
		if(t.pksum){
			pksum=std::make_unique<uint[]>(rg+1);
			std::memcpy(pksum.get(),t.pksum.get(),sizeof(uint)*(rg+1));
		}
		if(t.d){
			d=std::make_unique<uint[]>(rg+1);
			std::memcpy(d.get(),t.d.get(),sizeof(uint)*(rg+1));
		}
		if(t.ds){
			ds=std::make_unique<uint[]>(rg+1);
			std::memcpy(ds.get(),t.ds.get(),sizeof(uint)*(rg+1));
		}
		if(t.premu){
			premu=std::make_unique<ll[]>(rg+1);
			std::memcpy(premu.get(),t.premu.get(),sizeof(ll)*(rg+1));
		}
		if(t.preei){
			preei=std::make_unique<ll[]>(rg+1);
			std::memcpy(preei.get(),t.preei.get(),sizeof(ll)*(rg+1));
		}
		if(t.pred){
			pred=std::make_unique<ll[]>(rg+1);
			std::memcpy(pred.get(),t.pred.get(),sizeof(ll)*(rg+1));
		}
		if(t.preds){
			preds=std::make_unique<ll[]>(rg+1);
			std::memcpy(preds.get(),t.preds.get(),sizeof(ll)*(rg+1));
		}
	}
	std::vector<std::pair<uint,uint>> sieve::factor(uint k) const {
		std::vector<std::pair<uint,uint>> ret;
		while(k>1){
			uint p=ps[mnf[k]-1],c=0;
			while((k%p)==0) ++c,k/=p;
			ret.push_back({p,c});
		}
		return ret;
	}
	ull basic::gcdll(ull a,ull b){
		if(!a || !b) return a|b;uint t=__builtin_ctzll(a|b);
		a>>=__builtin_ctzll(a);
		do{
			b>>=__builtin_ctzll(b);
			if(a>b) std::swap(a,b);
			b-=a;
		}while(b);
		return a<<t;
	}
	uint basic::gcd(uint a,uint b){
		if(!a || !b) return a|b;uint t=__builtin_ctz(a|b);
		a>>=__builtin_ctz(a);
		do{
			b>>=__builtin_ctz(b);
			if(a>b) std::swap(a,b);
			b-=a;
		}while(b);
		return a<<t;
	}
	_random_engine random_engine(default_mod);
	ull factorization::fast_pow_mod(ull a,ull b,ull c){ull ans=1,off=a;while(b){if(b&1) ans=(__uint128_t)ans*off%c;off=(__uint128_t)off*off%c;b>>=1;}return ans;}
	bool factorization::is_prime(ull k){
		if(global_sieve_range>=k) return global_sieve.is_prime(k);
		for(ull c:bases) if(k==c) return true;
		for(ull c:bases) if(k>=c){
			ull d=k-1,res=fast_pow_mod(c,d,k);
			if(res!=1) return false;
			while(d&1^1){
				if((d>>=1),((res=fast_pow_mod(c,d,k))==k-1)) break;
				else if(res!=1){
					return false;
				}
			}
		}
		return true;
	}
	ull factorization::pollard_rho(ull x)
	{
		ull s=0,t=0,c=((ull)random_engine())%(x-1)+1,j=0,k=1,tmp=1;
		for(k=1;;k<<=1,s=t,tmp=1){
			for(j=1;j<=k;++j){
				t=((__uint128_t)t*t+c)%x;
				tmp=(__uint128_t)tmp*std::abs((ll)t-(ll)s)%x;
				if((j%127)==0){
					ull d=basic::gcdll(tmp,x);
					if(d>1) return d;
				}
			}
			ull d=basic::gcdll(tmp,x);
			if(d>1) return d;
		}
		return 1;
	}
	void factorization::_factorize(ull n,uint cnt,std::vector<ull> &pms){
		if(n<2) return;
		if(is_prime(n)){
			for(uint i=0;i<cnt;++i) pms.push_back(n);
			return;
		}
		ull p=n;
		while(p>=n) p=pollard_rho(n);uint c=0;
		while((n%p)==0) n/=p,++c;
		_factorize(n,cnt,pms),_factorize(p,cnt*c,pms);
	}
	std::vector<std::pair<ull,uint> > factorization::factor(ull k){
		if(global_sieve_range>=k){
			auto vv=global_sieve.factor(k);
			std::vector<std::pair<ull,uint>> v2;
			for(auto &&v:vv) v2.push_back(std::make_pair(v.first,v.second));
			return v2;
		}
		std::vector<ull> pms;_factorize(k,1,pms);
		std::sort(pms.begin(),pms.end());std::vector<std::pair<ull,uint>> res;
		for(uint i=0,j;i<pms.size();i=j){
			for(j=i;j<pms.size() && pms[j]==pms[i];++j);
			res.push_back(std::make_pair(pms[i],j-i));
		}
		return res;
	}
	polynomial_ntt::polynomial_ntt(uint max_conv_size,uint P0,uint G0){init(max_conv_size,P0,G0);}
	polynomial::polynomial(uint max_conv_size,uint P0):F1(P1),F2(P2),F3(P3){init(max_conv_size,P0);}
	linear_modulo_preprocessing::linear_modulo_preprocessing(uint maxn,uint P){init(maxn,P);}
	sieve::sieve(uint maxn,int flag){init(maxn,flag);}
	sieve global_sieve;uint global_sieve_range=0;int global_sieve_flag=0;
	void set_global_sieve(uint rg,int flag){
		global_sieve_range=rg;global_sieve_flag=flag;
		global_sieve.init(rg,flag);
	}
	ull factorization::fast_pow_without_mod(ull a,uint b){ull ans=1,off=a;while(b){if(b&1) ans=ans*off;off=off*off;b>>=1;}return ans;}
	std::vector<ull> factorization::divisors_set(const std::vector<std::pair<ull,uint>> &decomp){
		std::function<void(uint,ull,const std::vector<std::pair<ull,uint>> &,std::vector<ull> &)> calc=[&](uint pos,ull cur,const std::vector<std::pair<ull,uint>> &decomp,std::vector<ull> &res)->void{
			if(!pos){res.push_back(cur);return;}
			ull pp=1;
			for(uint i=0;i<=decomp[pos-1].second;++i){
				calc(pos-1,pp*cur,decomp,res);
				pp*=decomp[pos-1].first;
			}
		};
		std::vector<ull> res;
		calc(decomp.size(),1,decomp,res);
		std::sort(res.begin(),res.end());
		return res;
	}
	std::vector<ull> factorization::divisors_set(ull k){return divisors_set(factor(k));}
	int factorization::moebius(ull k){
		if(global_sieve_range>=k && (global_sieve_flag&sieve_mu)) return global_sieve.mu(k);
		return moebius(factor(k));
	}
	int factorization::moebius(const std::vector<std::pair<ull,uint>> &decomp){
		bool g2=false;
		for(auto &&v:decomp) if(v.second>=2){g2=true;break;}
		return g2?0:((decomp.size()&1)?(-1):1);
	}
	ull factorization::euler_phi(ull k){
		if(global_sieve_range>=k && (global_sieve_flag&sieve_euler_phi)) return global_sieve.euler_phi(k);
		return euler_phi(factor(k));
	}
	ull factorization::euler_phi(const std::vector<std::pair<ull,uint>> &decomp){
		ull ans=1;
		for(auto &&v:decomp) ans*=fast_pow_without_mod(v.first,v.second-1)*(v.first-1);
		return ans;
	}
	ull factorization::sigma(ull k,uint s){return sigma(factor(k),s);}
	ull factorization::sigma(const std::vector<std::pair<ull,uint>> &decomp,uint s){
		std::function<ull(ull,uint)> calc=[&](ull a,uint b){ull ans=0,off=1,sc=a;while(b){if(b&1) ans=ans*sc+off;off=off*sc+off;sc=sc*sc;b>>=1;}return ans;};
		ull ans=1;
		for(auto &&v:decomp) ans*=calc(fast_pow_without_mod(v.first,s),v.second+1);
		return ans;
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