'''
Test script for shared memory multi-processing in Python.

This script demonstrates how to use shared memory for multi-processing in Python,
using the `multiprocessing` module and `ctypes` for shared arrays.

It initializes a shared array, spawns multiple processes that modify the shared data,
and collects results from each process.

This is a simplified test script to verify that shared memory is working correctly.

A modified version is used in the EC-based line search implementations in `dmap_zncc.py`
and `dmap_daisy.py`, where the image data or descriptors are accessed from shared memory
to avoid unnecessary copying across processes.
'''

import ctypes
import numpy as np
import multiprocessing as mp
import sharedmem as sm
import func


if __name__ == '__main__':
    shared_shape = (10, 10)
    shared_size = shared_shape[0] * shared_shape[1]
    shared_base = mp.Array(ctypes.c_double, shared_size)
    
    # Initialize with random data
    init_data = np.ones(shared_shape)
    sm.init(shared_base, shared_shape)
    sm.shared_data[:] = init_data[:]
    
    # Create pool
    mp.freeze_support()
    pool = mp.Pool(processes = 4,
                   initializer = sm.init,
                   initargs = (shared_base, shared_shape))
    
    R = []
    for i in range(10):
        r = pool.apply_async(func.my_func, args=(i,))
        R.append(r)
    R = [r.get() for r in R]
    pool.close()
    pool.join()

    # Get value from shared data
    shared_data = np.ctypeslib.as_array(shared_base.get_obj())
    shared_data = shared_data.reshape(10, 10)
    print('Shared data')
    print(shared_data)
    
    # Print return values
    print('\nReturn data')
    print(R)
