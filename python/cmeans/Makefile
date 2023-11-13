GSL_INCLUDE = -I/usr/local/include -I/usr/include -L/usr/local/lib/gsl
GSL_FLAGS = -lgsl -O3 -lgslcblas -lm -march=native -ffast-math -Ofast -funroll-loops -fprefetch-loop-arrays

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Linux)
	CC = gcc
endif
ifeq ($(UNAME_S),Darwin)
	CC = clang
endif

ifeq ($(UNAME_M),arm64)
	GSL_FLAGS += -arch arm64
endif

compile: clean
	$(CC) $(GSL_FLAGS) $(GSL_INCLUDE)-g -Wall -c cmeans.c
	$(CC) $(GSL_FLAGS) $(GSL_INCLUDE) -o exe cmeans.o
	gtime -v ./exe


sharedlib: clean
	$(CC) $(GSL_INCLUDE) $(GSL_FLAGS) -Wall -fPIC -c cmeans.c
	$(CC) $(GSL_INCLUDE) $(GSL_FLAGS) -dynamiclib -o libcmeans.so cmeans.o


profile:
	valgrind --tool=callgrind ./exe


clean:
	rm -rf *.out *.dSYM exe *o *so *txt

# apt-get update && apt-get install -y libgsl-dev && gsl-config --version
docker:
	docker run -it -v `pwd`:/code ubuntu

check:
	cppcheck *.c