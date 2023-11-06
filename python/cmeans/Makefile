# on mac, brew install libgsl; on raspbian, sudo apt-get install libgsl0-dev
compile: clean
	gcc -g -I/usr/local/include -I/usr/include -Wall cmeans.c -o exe -lgsl -O3 -lgslcblas -lm && time ./exe
	#gcc -Wall -I/usr/include -c cmeans.c
	#gcc -L/usr/include cmeans.o -lgsl -lgslcblas -lmi


sharedlib: clean
	gcc -I/usr/local/include -I/usr/include -lgsl -O3 -lgslcblas -lm -Wall -fPIC -c cmeans.c
	gcc -I/usr/local/include -I/usr/include -lgsl -O3 -lgslcblas -lm -dynamiclib -o libcmeans.so cmeans.o


profile:
	valgrind --tool=callgrind ./exe


clean:
	rm -rf *.out *.dSYM exe *o *so *txt

# apt-get update && apt-get install -y libgsl-dev && gsl-config --version
docker:
	docker run -it -v `pwd`:/code ubuntu

check:
	cppcheck *.c