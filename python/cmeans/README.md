## Re-learning C

This is a toy implementation of k-means clustering in C using GSL. It's about 2.5x slower than sklearn but it has not been optimized at all (eg no BLAS, naive algorithm all around), so I secretly hope with some more work it can match (or even beat) sklearn.

### Installation

I want to write C, not a CI/CD pipeline. Install your own dependencies! eg `brew install gsl`, `pip install scikit-learn` etc

### Experiences

- C is enjoyable in a very primitive way. Makes you think about when memory is allocated, when it's freed, where intermediate values are saved, etc.
- C is so hard to write correctly because of silent and suprising behaviours. Division can be weird, the editors I'm familiar with (Intellij, VSCode) are not very useful, `memcpy` behaves weirdly for `uint8`, my vector averaging overflows and return `nan` but does not throw an error, reading from uninitialized memory, etc 