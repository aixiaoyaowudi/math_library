/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_TYPE_TYPE_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_TYPE_TYPE_H_

#include <cstdint>
#include <type_traits>
#include <memory>
#if defined(__INTEL_COMPILER)
#include <aligned_new>
#else
#include <new>
#endif
#include <modulo/modint.h>

namespace math
{
	template<typename T,size_t align_val>
	struct aligned_delete {
		void operator()(T* ptr) const {
			operator delete[](ptr,std::align_val_t(align_val));
		}
	};
	template<typename T,size_t align_val>
	using aligned_array=std::unique_ptr<T[],aligned_delete<T,align_val>>;
	template<typename T,size_t align_val>
	aligned_array<T,align_val> create_aligned_array(size_t size){
		return aligned_array<T,align_val>(new(std::align_val_t(align_val)) T[size]);
	}
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
			template<> struct size_int<8>:type_helper<std::int8_t>{};
			template<> struct size_int<16>:type_helper<std::int16_t>{};
			template<> struct size_int<32>:type_helper<std::int32_t>{};
			template<> struct size_int<64>:type_helper<std::int64_t>{};
			template<> struct size_int<128>:type_helper<__int128>{};
			template<size_t N> using size_int_t=typename size_int<N>::type;
			template<size_t N> struct size_uint:size_uint<N+1>{};
			template<> struct size_uint<8>:type_helper<std::uint8_t>{};
			template<> struct size_uint<16>:type_helper<std::uint16_t>{};
			template<> struct size_uint<32>:type_helper<std::uint32_t>{};
			template<> struct size_uint<64>:type_helper<std::uint64_t>{};
			template<> struct size_uint<128>:type_helper<__uint128_t>{};
			template<size_t N> using size_uint_t=typename size_uint<N>::type;
			template<typename T,typename std::enable_if<std::is_integral<T>::value>::type* =nullptr> struct double_width_int:size_int<2*sizeof(T)*8>{};
			template<typename T> using double_width_int_t=typename double_width_int<T>::type;
			template<typename T,typename std::enable_if<std::is_integral<T>::value>::type* =nullptr> struct double_width_uint:size_uint<2*sizeof(T)*8>{};
			template<typename T> using double_width_uint_t=typename double_width_uint<T>::type;
		}
	}
}

#endif