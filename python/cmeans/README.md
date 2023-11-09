# Setup for C  
on mac, brew install libgsl; on raspbian, sudo apt-get install libgsl0-dev
then (https://stackoverflow.com/a/38713916)
```
sudo ln -s /Users/mirobat/homebrew/Cellar/gsl/2.7.1/include/gsl /usr/local/include/gsl
sudo ln -s /Users/mirobat/homebrew/Cellar/gsl/2.7.1/lib /usr/local/lib/gsl
```
 
I saw a warning that OMP/BLAS are missing => `brew install libomp openblas` but that seems optional.

`make compile` to compile and executable and run it- useful for prototyping and profiling
`make sharedlib` to compile as a shared library for loading from python