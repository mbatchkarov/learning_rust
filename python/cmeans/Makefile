# on mac, brew install libgsl; on raspbian, sudo apt-get install libgsl0-dev
# then (https://stackoverflow.com/a/38713916)
# sudo ln -s /Users/mirobat/homebrew/Cellar/gsl/2.7.1/include/gsl /usr/local/include/gsl
# sudo ln -s /Users/mirobat/homebrew/Cellar/gsl/2.7.1/lib /usr/local/lib/gsl

GSL_INCLUDE = -I/usr/local/include -I/usr/include -L/usr/local/lib/gsl
GSL_FLAGS = -lgsl -O3 -lgslcblas -lm

compile: clean
	gcc -g $(GSL_INCLUDE) -Wall cmeans.c -o exe $(GSL_FLAGS) && time ./exe


sharedlib: clean
	gcc $(GSL_INCLUDE) $(GSL_FLAGS) -Wall -fPIC -c cmeans.c
	gcc $(GSL_INCLUDE) $(GSL_FLAGS) -dynamiclib -o libcmeans.so cmeans.o


profile:
	valgrind --tool=callgrind ./exe


clean:
	rm -rf *.out *.dSYM exe *o *so *txt

# apt-get update && apt-get install -y libgsl-dev && gsl-config --version
docker:
	docker run -it -v `pwd`:/code ubuntu

check:
	cppcheck *.c