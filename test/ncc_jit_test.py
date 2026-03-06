"""
Compare NCC computation times for Python, Numba, and Scipy implementations.

This script tests the performance of different implementations of the normalized
cross-correlation (NCC) function, including a pure Python version, a Numba-optimized
version with lazy compilation, a Numba-optimized version with nopython mode,
and a Numba-optimized version with eager compilation.

It also compares the performance of these implementations against the NCC computation
using SciPy's signal.correlate function.
"""

from __future__ import division
from __future__ import print_function
import cv2
import time
from numba import jit, float64, uint8
import numpy as np
from scipy import signal


# Python implementation of NCC (for reference)
def ncc_python(im1, im2):
    r1, c1 = im1.shape
    r2, c2 = im2.shape
    if (r1==r2) and (c1==c2) and (r1>0) and (c1>0):
        mean1 = np.mean(im1)
        mean2 = np.mean(im2)
        std1 = np.std(im1)
        std2 = np.std(im2)
        result = (im1-mean1) * (im2-mean2) / (std1 * std2)
        return np.round(np.sum(result)/(r1*c1), 4)
    else:
        return -1


# Numba-optimized NCC with lazy compilation (compiles on first call)
@jit
def ncc_numba_lazy(im1, im2):
    r1, c1 = im1.shape
    r2, c2 = im2.shape
    if (r1==r2) and (c1==c2) and (r1>0) and (c1>0):
        mean1 = np.mean(im1)
        mean2 = np.mean(im2)
        std1 = np.std(im1)
        std2 = np.std(im2)
        result = (im1-mean1) * (im2-mean2) / (std1 * std2)
        return np.round(np.sum(result)/(r1*c1), 4)
    else:
        return -1


# Numba-optimized NCC with nopython mode (compiles on first call, but runs in optimized mode)
@jit(nopython=True)
def ncc_numba_nopy(im1, im2):
    r1, c1 = im1.shape
    r2, c2 = im2.shape
    if (r1==r2) and (c1==c2) and (r1>0) and (c1>0):
        mean1 = np.mean(im1)
        mean2 = np.mean(im2)
        std1 = np.std(im1)
        std2 = np.std(im2)
        result = (im1-mean1) * (im2-mean2) / (std1 * std2)
        return np.round(np.sum(result)/(r1*c1), 4)
    else:
        return -1


# Numba-optimized NCC with eager compilation (compiles at definition time, runs in optimized mode)
@jit(float64(uint8[:,:],uint8[:,:]))
def ncc_numba_eager(im1, im2):
    r1, c1 = im1.shape
    r2, c2 = im2.shape
    if (r1==r2) and (c1==c2) and (r1>0) and (c1>0):
        mean1 = np.mean(im1)
        mean2 = np.mean(im2)
        std1 = np.std(im1)
        std2 = np.std(im2)
        result = (im1-mean1) * (im2-mean2) / (std1 * std2)
        return np.round(np.sum(result)/(r1*c1), 4)
    else:
        return -1


if __name__ == "__main__":
    # Test image
    impath = 'data\\Dataset\\Dinosaur\\images\\viff.000.ppm'
    I = cv2.imread(impath, 0)

    # Extract two patches
    p0 = I[120:220,120:220]
    p1 = I[150:250,120:220]
    
    # Run each implementation for different numbers of iterations and measure time taken
    N = [1, 100, 10000]
    
    for niter in N:
        stime = time.time()
        for i in range(niter):
            score = ncc_python(p0, p1)
        etime = time.time() - stime
        print("{:16} {:5} run: {:.2f} s".format('ncc_python',niter,etime))
        
        stime = time.time()
        for i in range(niter):
            score = ncc_numba_lazy(p0, p1)
        etime = time.time() - stime
        print("{:16} {:5} run: {:.2f} s".format('ncc_numba_lazy',niter,etime))
        
        stime = time.time()
        for i in range(niter):
            score = ncc_numba_nopy(p0, p1)
        etime = time.time() - stime
        print("{:16} {:5} run: {:.2f} s".format('ncc_numba_nopy',niter,etime))
        
        stime = time.time()
        for i in range(niter):
            score = ncc_numba_eager(p0, p1)
        etime = time.time() - stime
        print("{:16} {:5} run: {:.2f} s".format('ncc_numba_eager',niter,etime))
        
        stime = time.time()
        for i in range(niter):
            # np.uint8 type casting to avoid type conversion that would break
            # compatibility with numba eager function definition
            p0 = np.uint8(p0 - p0.mean())
            p1 = np.uint8(p1 - p1.mean())
            score = signal.correlate(p0, p1, mode='valid')
        etime = time.time() - stime
        print("{:16} {:5} run: {:.2f} s".format('scipy-correlate',niter,etime))
        print()
