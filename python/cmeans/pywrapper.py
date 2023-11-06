# run `make sharedlib` first
from os.path import join, dirname
import ctypes
import numpy
import numpy as np
from numpy.ctypeslib import ndpointer

lib = ctypes.cdll.LoadLibrary(join(dirname(__file__), "libcmeans.so"))
_cluster = lib.cluster
_cluster.restype = None
_cluster.argtypes = [
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS"),
]


def cluster(data: np.array, k):
    nrows, ncols = data.shape
    outdata = numpy.empty((nrows,), dtype=np.uint8)
    _cluster(data, *data.shape, k, outdata)
    return outdata


if __name__ == "__main__":
    # now call the bad boy
    indata = numpy.random.random((10, 3))
    print(cluster(indata, 3))
