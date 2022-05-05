COMPILER     = icpc
CPPFLAGS     = -std=c++17 -Ofast -I./ -march=coffeelake -fopenmp -restrict
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

bin/test:func_tests.cpp all
	$(COMPILER) $(CPPFLAGS) $< -L./bin/ -lpe -o bin/test

test:bin/test
	./bin/test

all: bin/pe.hpp bin/coefs_for_fast_binomial_2_32 bin/coefs_for_fast_binomial_2_64 bin/libpe.a
	@echo done >/dev/null

clean:
	@rm -rf bin/