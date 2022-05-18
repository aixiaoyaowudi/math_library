/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_BINOMIAL_LINEAR_MODULO_PREPROCESSING_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_BINOMIAL_LINEAR_MODULO_PREPROCESSING_H_

#include <type/basic_typedef.h>
#include <memory>
#include <modulo/modint.h>

namespace math
{
	namespace binomial
	{
		class linear_modulo_preprocessing
		{
		private:
			std::unique_ptr<mi[]> fac,ifac,_inv;ui rg,P;
			void release();
		public:
			linear_modulo_preprocessing();
			linear_modulo_preprocessing(const linear_modulo_preprocessing &d);
			~linear_modulo_preprocessing();
			void init(ui maxn,ui P);
			#define INLINE_OP __attribute__((__always_inline__))
			INLINE_OP mi factorial(ui i){return fac[i];}
			INLINE_OP mi inverse_factorial(ui i){return ifac[i];}
			INLINE_OP mi inverse(ui i){return _inv[i];}
			INLINE_OP mi binomial(ui upper,ui lower){if(lower>upper) return mi(0);else return fac[upper]*ifac[lower]*ifac[upper-lower];}
			#undef INLINE_OP
		};
	}
}

#endif