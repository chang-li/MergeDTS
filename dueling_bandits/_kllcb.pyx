cimport cython
import numpy as np
cimport numpy as np
import sys

from libc.math cimport log as c_log
from libc.math cimport abs as c_abs
from libc.stdio cimport printf

ctypedef np.float64_t DOUBLE
ctypedef np.int32_t INT

np.import_array()

cdef double DELTA = 1e-7
cdef double GAMMA = 1e-6

cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b


cdef double KLBernoulli(double p, double q):
    if p > 1 or q > 1:
        return -1
    if p <= DELTA:
        return - c_log(1 - q)
    if p >= 1-DELTA:
        return - c_log(q)
    return p * c_log(p / q) + (1 - p) * c_log((1 - p) / (1 - q))

cpdef np.ndarray[DOUBLE, ndim=1] kllcb(
        np.ndarray[DOUBLE, ndim=1] gains,
        np.ndarray[DOUBLE, ndim=1] N_plays,
        int n,
        int delta,
        double precision):
    cdef:
        Py_ssize_t i
        Py_ssize_t j
        double upper_bound = 0
        double u_m = 0
        double u_M = 0
        double f_max = 0
        double p = 0
        cdef Py_ssize_t K = gains.shape[0]
        np.ndarray[DOUBLE, ndim=1] res = np.zeros(
                K, dtype=np.float64)
    for i in range(K):
        upper_bound = (c_log(n) + delta * c_log(c_log(n))) / N_plays[i]
        p = gains[i] / N_plays[i]
        if p <= 0 + DELTA:
            res[i] = 0
            continue
        u_m = DELTA
        u_M = p
        f_max = upper_bound - KLBernoulli(p, u_m)
        if u_M <= DELTA or f_max >= 0:
            res[i] = 0
            continue
        j = 0
        while (u_M - u_m) > precision:
            if (upper_bound - KLBernoulli(p, (u_m + u_M) / 2)) >= 0:
                u_M = (u_m + u_M) / 2
            else:
                u_m = (u_m + u_M) / 2
        res[i] = (u_m + u_M) / 2
    return res
