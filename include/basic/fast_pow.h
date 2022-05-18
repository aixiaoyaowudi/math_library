/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_BASIC_FAST_POW_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_BASIC_FAST_POW_H_

#include <type/type.h>

namespace math
{
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
}

#endif