"""
Compute dense depth-maps using 1D evolutionary search with ZNCC patch similarity.

This script drives the EC-based line search implemented in `modules/ec.py` to
compute per-pixel depth values for images in a multi-view dataset.

Supports:
- Single-core execution via `ec.line_search()`
- Shared-memory multi-processing via `ec.line_search_shm()`

The EC algorithm can be changed at ec.line_search() (default is CMA-ES which performs well).
"""

import os
import time
import numpy as np
import scipy.io as spio
from tqdm import tqdm

import modules.func as func
import modules.ec as ec


# Single core or parallel execution
is_parallel = True

# Image dataset and resolution (width, height)
dataset = 'dinoRing'
res = (640, 480)

# Path for data
data_path = 'data'

# EC algorithm parameters (population size, number of generations, depth bounds)
n_pop = 20
n_gen = 30
B = [0.5, 0.8]

# Camera parameters
ang_bound = [5, 35]          # Angles for neighbor camera selection [degrees]
k_best = 5                   # Fitness aggregation of best `k` neighbors [1, np.inf]
# k_best = 0.7                # Fitness aggregation of best `k%` neighbors [0, 0.99]

# ZNCC patch size (must be odd for symmetry)
patch_size = 3


# Progress bar update callback for parallel processing
def pbar_update(p):
    pbar.update()


if __name__ == '__main__':
    print('')
    print('********************************************************************')
    print('Find depth for each pixel using 1D EC search with ZNCC as similarity')
    print('********************************************************************')
    print('')
    print('Dataset                  : {}'.format(dataset))
    print('Resolution (w x h)       : {} x {}'.format(res[0], res[1]))
    print('Depth bounds             : [{}, {}]'.format(B[0], B[1]))
    print('Angular bounds           : [{}, {}]'.format(ang_bound[0], ang_bound[1]))
    print('Fitness of best k        : {}'.format(k_best))
    print('EC params                : n_pop = {}, n_gen = {}'.format(n_pop, n_gen))
    print('ZNCC patch-size          : {}'.format(patch_size))
    print('')

    # Load images and camera matrices
    im_dir = os.path.join(data_path, dataset, 'images')
    I = func.load_images_from_folder(im_dir, 0)
    mat_file_path = os.path.join(data_path, dataset, dataset + '_P.mat')
    mat_file = spio.loadmat(mat_file_path)
    P = mat_file['P']
    P = np.asarray(P, order='C')

    # Check if camera matrices for all images are available
    n_cam = len(P)
    if n_cam != len(I):
        raise ValueError('n_cam n_images mismatch')

    # Camera layout (position, orientation, angle) for each camera
    POS, ORN, ANG = func.cam_layout(P)

    # Multi-processing initialisation
    if (is_parallel):
        import multiprocessing as mp
        import ctypes
        import modules.sharedmem as sm

        shared_size = I.size
        shared_shape = I.shape
        shared_base = mp.Array(ctypes.c_double, shared_size)
        sm.init(shared_base, shared_shape)
        sm.shared_data[:] = I[:]

        mp.freeze_support()                         # Required for Windows systems
        NPROC = mp.cpu_count() - 4                  # Number of processes to spawn
        if (NPROC < 2):                             # Set minimum 2
            NPROC = 2

        R = []
        pool = mp.Pool(processes = NPROC,
                       initializer = sm.init,
                       initargs = (shared_base, shared_shape))
    else:
        # Single-core execution (no shared memory, direct calls to ec.line_search)
        NPROC = 1

    # Initialize lists for depth, fitness, and pixel coordinates
    TA = []
    FA = []
    PA = []

    # Calculate border size based on patch size
    border = patch_size // 2

    stime = time.time()
    im_ind = np.arange(0, n_cam, dtype=np.int16)                 # Images to process
    x_ind = np.arange(border, res[0]-border, dtype=np.int16)
    y_ind = np.arange(border, res[1]-border, dtype=np.int16)
    combined = [(r,s,t) for r in im_ind for s in x_ind for t in y_ind]
    n_pts = len(combined)
    
    print('Processing {:,} points on {} threads...'.format(n_pts, NPROC))
    pbar = tqdm(total=n_pts, unit=' points')

    # Process pixels
    for i, iv in enumerate(combined):
        imx0 = iv[0]
        x = iv[1]
        y = iv[2]
        PA.append([imx0, x, y])

        if (is_parallel):
            r = pool.apply_async(ec.line_search_shm,
                                 args = (P, B, ANG[imx0], imx0, x, y, patch_size, n_pop, n_gen,
                                         ang_bound, k_best),
                                 callback = pbar_update)
            R.append(r)

        else:
            out = ec.line_search(I, P, B, ANG[imx0], imx0, x, y, patch_size, n_pop, n_gen,
                                 ang_bound, k_best)
            pbar.update()
            TA.append(out[0][0])
            FA.append(out[1][0])

    if (is_parallel):
        R = [r.get() for r in R]
        pool.close()
        pool.join()
        TA = [r[0][0] for r in R]
        FA = [r[1][0] for r in R]

    pbar.close()

    # Convert data to np.float32 to save disk space
    TA = np.float32(TA)
    FA = np.float32(FA)

    # Save final results
    rand_int = np.random.randint(100000, 999999)
    file_name = dataset + '_dense_' + str(rand_int) + '.mat'
    spio.savemat(file_name, dict(TA=TA, FA=FA, PA=PA), do_compression=True)

    # Print stats
    ttime = time.time() - stime
    print(' ')
    print('Total time           : {:.4} s'.format(ttime))
    print('Total points         : {}'.format(n_pts))
    print('Time taken per point : {:.4} ms'.format(ttime*1000/n_pts))
