"""
Compute dense depth-maps using 1D evolutionary search with OpenCV-DAISY feature matching.

This script drives the EC-based line search implemented in `modules/ec.py` to
compute per-pixel depth values for images in a multi-view dataset using OpenCV-DAISY
descriptors for feature matching.

Supports:
- Single-core execution via `ec.line_search_daisy()`
- Shared-memory multi-processing via `ec.line_search_daisy_shm()`

The EC algorithm can be changed at ec.line_search() (default is CMA-ES which performs well).

The results are saved in .mat format containing:
- TA: Depth values for each pixel
- FA: Fitness values for each pixel
- PA: Pixel coordinates (image index, x, y) for each depth value
"""

import os
import time
import numpy as np
import scipy.io as spio
import cv2
from tqdm import tqdm

import modules.func as func
import modules.ec as ec


# Single core or Parallel execution
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

# Daisy parameters (check OpenCV-DAISY documentation for details)
daisy_step = 1
daisy_radius = 3
daisy_rings = 1
daisy_histograms = 4
daisy_orientations = 4
daisy_norm = 102            # 100=None, 101=Partial, 102=Full, 103=Sift


# Progress bar update callback for parallel processing
def pbar_update(p):
    pbar.update()


if __name__ == '__main__':
    print('')
    print('****************************************************************************')
    print('Find depth for each pixel using 1D EC search with OpenCV-DAISY as similarity')
    print('****************************************************************************')
    print('')
    print('Dataset                  : {}'.format(dataset))
    print('Resolution (w x h)       : {} x {}'.format(res[0], res[1]))
    print('Depth bounds             : [{}, {}]'.format(B[0], B[1]))
    print('Angular bounds           : [{}, {}]'.format(ang_bound[0], ang_bound[1]))
    print('Fitness of best k        : {}'.format(k_best))
    print('EC params                : n_pop = {}, n_gen = {}'.format(n_pop, n_gen))
    print('DAISY params             : {}, {}, {}, {}, {}, {}'.format(daisy_step, daisy_radius,
                                                                     daisy_rings, daisy_histograms,
                                                                     daisy_orientations, daisy_norm))
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

    # Set OpenCV DAISY descriptor parameters
    daisy = cv2.xfeatures2d.DAISY_create(radius = daisy_radius,
                                         q_radius = daisy_rings,
                                         q_theta = daisy_histograms,
                                         q_hist = daisy_orientations,
                                         norm = daisy_norm)

    # Compute Daisy descriptors for each image
    D = []
    print('Computing cv2.DAISY descriptors...')
    for i in tqdm(range(n_cam)):
        kp = [cv2.KeyPoint(x, y, -1) for y in range(0, I[i].shape[0], 1)
                                            for x in range(0, I[i].shape[1], 1)]
        kp2, des = daisy.compute(I[i], kp)
        des = np.reshape(des, I[i].shape + (des.shape[1],))
        D.append(des)
    D = np.asarray(D, dtype=np.float32)

    # Multi-processing initialisation
    if (is_parallel):
        import multiprocessing as mp
        import ctypes
        import modules.sharedmem as sm

        shared_size = D.size
        shared_shape = D.shape
        shared_base = mp.Array(ctypes.c_double, shared_size)
        sm.init(shared_base, shared_shape)
        sm.shared_data[:] = D[:]

        mp.freeze_support()                 # Required for Windows systems
        n_proc = mp.cpu_count() - 2         # Leave some cores free to avoid system overload
        if (n_proc < 2):                    # Ensure at least 2 processes for parallel execution
            n_proc = 2

        R = []
        pool = mp.Pool(processes = n_proc,
                       initializer = sm.init,
                       initargs = (shared_base, shared_shape))
    else:
        # Single-core execution (no shared memory, direct calls to ec.line_search_daisy)
        n_proc = 1

    # Initialize lists for depth, fitness, and pixel coordinates
    TA = []
    FA = []
    PA = []

    # Calculate border size based on daisy radius
    border = int(daisy_radius)

    stime = time.time()
    im_ind = np.arange(0, n_cam, dtype=np.int16)                # Images to process
    x_ind = np.arange(border, res[0]-border, dtype=np.int16)
    y_ind = np.arange(border, res[1]-border, dtype=np.int16)
    combined = [(r,s,t) for r in im_ind for s in x_ind for t in y_ind]
    n_pts = len(combined)

    print('Processing {:,} points on {} threads...'.format(n_pts, n_proc))
    pbar = tqdm(total=n_pts, unit=' points')

    # Process pixels
    for i, iv in enumerate(combined):
        imx0 = iv[0]
        x = iv[1]
        y = iv[2]
        PA.append([imx0, x, y])

        if (is_parallel):
            r = pool.apply_async(ec.line_search_daisy_shm,
                                 args = (P, B, ANG[imx0], imx0, x, y, n_pop, n_gen, ang_bound, k_best),
                                 callback = pbar_update)
            R.append(r)
        else:
            out = ec.line_search_daisy(D, P, B, ANG[imx0], imx0, x, y, n_pop, n_gen, k_best)
            pbar.update()
            TA.append(out[0][0])
            FA.append(out[1][0])

    if (is_parallel):
        R = [r.get() for r in R]
        pool.close()
        pool.join()
        TA = [r[0][0] for r in R]
        FA = [r[1][0] for r in R]

    # Close progress bar
    pbar.close()

    # Convert data to np.float32 to save disk space
    TA = np.float32(TA)
    FA = np.float32(FA)

    # Save final results
    rand_int = np.random.randint(100000, 999999)
    file_name = dataset + '_dmaps_' + str(rand_int) + '.mat'
    spio.savemat(file_name, dict(TA=TA, FA=FA, PA=PA), do_compression=True)

    # Print stats
    ttime = time.time() - stime
    print(' ')
    print('Total time           : {:.4} s'.format(ttime))
    print('Total points         : {}'.format(n_pts))
    print('Time taken per point : {:.4} ms'.format(ttime*1000/n_pts))
