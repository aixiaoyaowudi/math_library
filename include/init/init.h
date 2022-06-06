/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#ifndef _XIAOYAOWUDI_MATH_LIBRARY_INIT_INIT_H_
#define _XIAOYAOWUDI_MATH_LIBRARY_INIT_INIT_H_

namespace math
{
	namespace init
	{
		__attribute__((constructor)) void math_library_init();
		#if defined(__INTEL_COMPILER)
		__asm__ __volatile__(".global _ZN4math4init17math_library_initEv");
		#else 
		typedef void (*__math_library_init_function_dummy_pointer_type)();
		static volatile __math_library_init_function_dummy_pointer_type __math_library_init_function_dummy_pointer=math_library_init;
		#endif
	}
}

#endif