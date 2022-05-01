COMPILER     = icpc
CPPFLAGS     = -std=c++17 -Ofast -I./
OPENMP_FLAGS = -fopenmp
AVX_FLAGS    = -xcore-avx2
PACK         = ar -crv

default: all
	@echo done >/dev/null

bin/pe.hpp:pe.hpp
	@mkdir -p bin/
	cp $^ bin/pe.hpp

bin/coefs_for_fast_binomial_2_64:coefs_for_fast_binomial_2_64
	@mkdir -p bin/
	cp $^ bin/coefs_for_fast_binomial_2_64

bin/coefs_for_fast_binomial_2_32:coefs_for_fast_binomial_2_32
	@mkdir -p bin/
	cp $^ bin/coefs_for_fast_binomial_2_32

bin/libpe.a:pe.cpp pe.hpp
	@mkdir -p bin/
	$(COMPILER) $(CPPFLAGS) $< -c -o bin/libpe.o
	$(PACK) bin/libpe.a bin/libpe.o
	@rm -f bin/libpe.o

bin/libpe_with_omp.a:pe.cpp pe.hpp
	@mkdir -p bin/
	$(COMPILER) $(CPPFLAGS) $(OPENMP_FLAGS) $< -c -o bin/libpe_with_omp.o
	$(PACK) bin/libpe_with_omp.a bin/libpe_with_omp.o
	@rm -f bin/libpe_with_omp.o

bin/libpe_with_avx2.a:pe.cpp pe.hpp
	@mkdir -p bin/
	$(COMPILER) $(CPPFLAGS) $(AVX_FLAGS) $< -c -o bin/libpe_with_avx2.o
	$(PACK) bin/libpe_with_avx2.a bin/libpe_with_avx2.o
	@rm -f bin/libpe_with_avx2.o

bin/libpe_with_omp_avx2.a:pe.cpp pe.hpp
	@mkdir -p bin/
	$(COMPILER) $(CPPFLAGS) $(AVX_FLAGS) $(OPENMP_FLAGS) $< -c -o bin/libpe_with_omp_avx2.o
	$(PACK) bin/libpe_with_omp_avx2.a bin/libpe_with_omp_avx2.o
	@rm -f bin/libpe_with_omp_avx2.o

all: bin/pe.hpp bin/coefs_for_fast_binomial_2_32 bin/coefs_for_fast_binomial_2_64 bin/libpe.a bin/libpe_with_omp.a bin/libpe_with_avx2.a bin/libpe_with_omp_avx2.a
	@echo done >/dev/null

clean:
	@rm -rf bin/