'''
Numba Performance Test Script

This script tests the performance of Numba JIT compilation in different modes
(lazy, nopython, eager) for a simple function that sums a large array.

The goal is to demonstrate the potential speedup from using Numba and the
differences between compilation modes.      
'''

import time
from numba import jit, float64


# Without Numba (baseline performance)
def test_0(x):
    start_time = time.time()
    out = np.sum(x)
    end_time = time.time() -start_time
    print('\ntest_0 "Without Numba"')
    print('Elapsed time: {:.4} s'.format(end_time))
    return out


# Lazy compilation (default mode, compiles on first call)
def test_1(x):   
    @jit
    def foo(x):
        return np.sum(x)
    
    start_time = time.time()
    out = foo(x)
    end_time = time.time() -start_time
    print('\ntest_1 "Lazy compilation"')
    print('Elapsed time: {:.4} s'.format(end_time))
    return out


# nopython mode (compiles to machine code, no Python objects allowed, best performance)
def test_2(x):   
    @jit(nopython=True)
    def foo(x):
        return np.sum(x)
    
    start_time = time.time()
    out = foo(x)
    end_time = time.time() -start_time
    print('\ntest_2 "nopython=True"')
    print('Elapsed time: {:.4} s'.format(end_time))
    return out


# Eager compilation (specify signature, compiles immediately, useful for testing
# and when input types are known)
def test_3(x):   
    @jit(float64(float64[:,:]))
    def foo(x):
        return np.sum(x)
    
    start_time = time.time()
    out = foo(x)
    end_time = time.time() -start_time
    print('\ntest_3 "Eager compilation"')
    print('Elapsed time: {:.4} s'.format(end_time))
    return out


if __name__ == '__main__':
    import numpy as np
    
    P = np.random.rand(1,1000000)           # Random test data
    out1 = test_0(P)
    out1 = test_1(P)
    out2 = test_2(P)
    out3 = test_3(P)
