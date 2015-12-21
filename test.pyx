import numpy as np
cimport numpy as np


DTYPE = np.int
ctypedef np.int_t DTYPE_t


def test(np.ndarray[DTYPE_t, ndim=2] a):
    a[0,0] = 2


def main():
    cdef np.ndarray[DTYPE_t, ndim=2] aa = np.array([[1,2,3],[4,5,6]])
    print aa
    test(aa)
    print aa
