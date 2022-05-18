/**
* @author Ziyao Xiao
* @mail   aixiaoyaowudi@gmail.com
**/

#include <modulo/modint.h>
#if defined(_OPENMP)
#include <omp.h>
#endif

namespace math
{
	namespace modulo{
		namespace modint{
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
			#if defined(__AVX512F__) && defined(__AVX512DQ__)
			ui global_mod_m5i=default_mod;
			montgomery_mm512_lib m5i::mlib(default_mod);
			void set_mod_m5i(ui p){
				m5i::mlib=montgomery_mm512_lib(p);
				global_mod_m5i=p;
			}
			void set_mod_for_all_threads_m5i(ui p){
				#if defined(_OPENMP)
				#pragma omp parallel
				{
				#endif
					m5i::mlib=montgomery_mm512_lib(p);
					global_mod_m5i=p;
				#if defined(_OPENMP)
				}
				#endif
			}
			#endif
		}
	}
}