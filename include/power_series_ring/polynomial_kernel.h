/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_POWER_SERIES_RING_POLYNOMIAL_KERNEL_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_POWER_SERIES_RING_POLYNOMIAL_KERNEL_H_

#include <modulo/modint.h>
#include <type/basic_typedef.h>
#include <type/type.h>
#include <vector>
#include <array>

namespace math
{
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
				aligned_array<ui,64> ws0,ws1,_inv,tt[tmp_size],num;ui P,G;
				ui fn,fb,mx;
				void release();
				ui _fastpow(ui a,ui b);
				void dif(ui* restrict arr,ui n);
				void dit(ui* restrict arr,ui n,bool last_layer=true);
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
				#if defined(__AVX512F__) && defined(__AVX512DQ__)
				lm5 l5;
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

#endif