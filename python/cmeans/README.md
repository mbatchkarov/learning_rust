## Re-learning C

This is a toy implementation of k-means clustering in C using GSL. It's about 2.5x slower than sklearn but it has not been optimized at all (eg no BLAS, naive algorithm all around), so I secretly hope with some more work it can match (or even beat) sklearn.

### Installation

I want to write C, not a CI/CD pipeline. Install your own dependencies! eg `brew install gsl`, `pip install scikit-learn` etc