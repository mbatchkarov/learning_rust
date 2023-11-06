# run `make sharedlib` first
import ctypes
import numpy
import numpy as np
from numpy.ctypeslib import ndpointer

lib = ctypes.cdll.LoadLibrary("./libcmeans.so")
cluster = lib.cluster
cluster.restype = None
cluster.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ndpointer(ctypes.c_uint8, flags="C_CONTIGUOUS")]

# now call the bad boy
nrows = 6
indata = numpy.random.random((nrows, 3))
outdata = numpy.empty((nrows,), dtype=np.uint8)
print(indata.size)
cluster(indata, *indata.shape, 3, outdata)
print(outdata, type(outdata))
