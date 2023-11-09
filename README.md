## Re-learning C

This is a toy implementation of k-means clustering in C using GSL, Rust using ndarray, and Python using numpy. This repo is entirely for educational purposes - I have no experience with Rust and haven't touched C in 10 years. My aim is to get better at Rust/C and get a feel for the tradeoffs between "ergonomics" and runtime speed

### Installation

Things are semi-manual at the moment
C: 
Rust: install rust
I want to write C, not a CI/CD pipeline. Install your own dependencies! eg `brew install gsl`, `pip install scikit-learn` etc

### Results

Speed comparison for 10k x 1.5k matrix w/ 10 clusters (M1 pro macbook pro, 2015 Intel macbook pro- clang14):

|         | Time, s (Intel) | Time, s (M1) | Memory, MB | Days to write |
|---------|-----------------|--------------|------------|---------------|
| Numpy   | 0.45            | 0.24         | 450        | 0.2           | 
| C       | 1.60            | 1.61         | 372        | 2             |
| Rust    | 0.13            | 0.08         | 368        | 2             |
| Sklearn | 0.20            | 0.10         | 487        | -             |

Notes:
- M1 is 3-4x faster than a previous-gen intel mac, nice. I must be doing something wrong with the compilation of the C version on M1 because it can't beat the intel mac.
- Flamegraph shows most of the time is spent in euclidean distance calculation (map a->a*a, subtract, sum), can optimize there
  - rewrote rust implementation of euclidean distance, got 2x speedup
  - used `rayon` to compute distances in parallel in rust- got 2x speedup for 30 min of work
  - used sklearn's pairwise distance function to get 1) parallel computation and 2) an optimized distance function from scipy. This was a 5x boost for 5 min of work.

Method for memory use: edit `test_all.py` to leave only one clustering function, then run 
```bash
/usr/local/bin/gtime -v pytest -s
```

Method for runtime: see `speed.ipynb` notebook in this repo.


### Experiences

- C is enjoyable in a very primitive way. Makes you think about when memory is allocated, when it's freed, where intermediate values are saved, etc.
- C is so hard to write correctly because of silent and suprising behaviours. It makes you think you have a working piece of code but there are in fact many pitfalls and silent failures that take a experience and time to resolve. Examples: division can be weird, the editors I'm familiar with (Intellij, VSCode) are not very useful, `memcpy` behaves weirdly for `uint8`, my vector averaging overflows and return `nan` but does not throw an error, reading from uninitialized memory, etc

- Rust: makes it easy to optimize (concurrency, flamegraph, rewrite hot func)
- running in parallel makes the flame graph huge, very hard to tell what takes time

# Python notes

- I had a working translation of the rust version in 30 min
- started without mypy because typing np shapes isn't super easy
- encountered multiple errors that could have been caught by a type checker:
  - positional function arguments were swapped, caught at first test run
  - index err at runtime, state mutated accidentally to the wrong shape (must be mutable though). I took argmin along the wrong axis, producing wrong array shape- no warning)
  - I didn't specify an axis when averaging clusters, got a scalar instead of vector
- A good IDE is essential. I stated off in VSCode because I want to like it, but I get so little assistance from it I immediately switched back to PyCharm. 

It took me a total or 5 runs of the unit test, 10-15 min to fix all the above. I'm on the fence whether python or rust version would be easier to maintain. I find the python one easier to read because it's shorter and because I'm much more experienced with numpy slicing- but maybe there's too much magic and others may find this harder. With a small investment into typing I could probably get something that's self-documenting like rust

The quality of the python scientific ecosystem blows me away every time, even after 10 year. The fact that everyone is using numpy, scipy and sklearn means these work very well together, so it's (usually) easy to benefit from well-optimized low-level code. Eg in sklearn, set `dist='sqeuclidean'` to use a good implementation of a distance function from scipy that takes numpy arrays as input. You get 90% of the way there with 10% of the effort. To get the same in Rust, I had to either write my own (not as fast) or google for less-known libraries
  - https://github.com/lazear/simd-euclidean
  - https://blog.lancedb.com/my-simd-is-faster-than-yours-fb2989bf25e7

### Usage instructions
Python: install `requirements.txt`
C: see Readme in `./cmeans` directory
Rust: install rust; `pip install maturin`; `maturin build -r` to build python bindings in release mode (much faster than dev mode); `cargo install flamegraph && sudo cargo flamegraph` for profiling

### TODOs

- for numpy version, use `sklearn.pairwise` to compute distances --> runtime goes from 1.2s to 0.38s (with 1 job) and 0.25s (with n_jobs == N_cpu)
- Bug to fix: python version sometimes crashes due to division by zero when averaging the data points in a cluster (when the cluster is empty). This bug likey exists in the  C/Rust version too but I didn't see any warnings or errors
- profile C/Rust some more and try to vectorize the tight inner loop (distance between all centroids and data points)