"""
Generic utility functions for multi-view depth estimation experiments.

This module collects camera geometry helpers, image patch utilities, similarity
measures (ZNCC), gradient and filtering routines, file-format converters, and
cross-view consistency checks. Many functions are Numba-jitted for speed and
are used by the higher-level scripts such as ``dmap_zncc.py`` and
``dmap_daisy.py``.
"""

import os
import gc
import sys
import cv2
import numpy as np
from numba import jit
import scipy.linalg as spla
import numpy.linalg as npla


# Taken from: https://stackoverflow.com/a/53705610/7046003
# Returns size of object in bytes
def get_obj_size(obj):
    marked = {id(obj)}
    obj_q = [obj]
    sz = 0

    while obj_q:
        sz += sum(map(sys.getsizeof, obj_q))

        # Lookup all the object referred to by the object in obj_q.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

        # Filter object that are already marked.
        # Using dict notation will prevent repeated objects.
        new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

        # The new obj_q will be the ones that were not marked,
        # and we will update marked with their ids so we will
        # not traverse them again.
        obj_q = new_refr.values()
        marked.update(new_refr.keys())

    return sz


# Load images from a folder
# mode: 0 = Grayscale, 1 = Color, -1 = Unchanged
# is_return_file_list: If True, return a list of file names in addition to the images
def load_images_from_folder(folder, mode=0, is_return_file_list=False):
    images = []
    file_list = []

    for file_name in sorted(os.listdir(folder)):
        file_list.append(file_name)
        img = cv2.imread(os.path.join(folder, file_name), mode)

        if img is not None:
            if img.ndim > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # Convert BGR to RGB
            images.append(img)
    images = np.asarray(images)

    if (is_return_file_list):
        return images, file_list
    else:
        return images


# Return a patch of size psize x psize from image 'im' centered at (x,y).
# If the patch extends beyond the image borders, return a patch filled with NaN values.
@jit(nopython=True, nogil=True)
def get_patch(im, x, y, psize):
    d = psize // 2
    x = np.int_(np.round(x))
    y = np.int_(np.round(y))
    xl = x - d
    xh = x + d + 1
    yl = y - d
    yh = y + d + 1

    if (xl<0) or (yl<0) or (xh>im.shape[1]) or (yh>im.shape[0]):
        return np.full((psize,psize), np.nan, dtype=np.float32)
    else:
        return im[yl:yh, xl:xh].astype(np.float32)


# Compute camera parameters (K, R, T) from camera matrix
def cam_factor(P):
    K, R = spla.rq(P[:,0:3])
    S = np.diag(np.sign(np.diag(K)))
    K = np.dot(K, S)
    R = np.dot(S, R)
    T = np.dot(spla.inv(K), P[:,3])
    T = T.reshape((3,1))
    return K, R, T


def cam_position(P):
    K, R, T = cam_factor(P)
    return -np.transpose(R).dot(T)


def cam_position_all(P):
    ncam = len(P)
    POS = np.zeros((ncam,3,1))

    for i in range(ncam):
        POS[i] = cam_position(P[i])
    return POS


def cam_orientation(P):
    o = P[-1,0:3].reshape((3,1))
    if npla.norm(o) != 0:
        o = o / npla.norm(o)
    return o


def cam_orientation_all(P):
    ncam = len(P)
    ORN = np.zeros((ncam,3,1))

    for i in range(ncam):
        ORN[i] = cam_orientation(P[i])
    return ORN


# Camera ray from camera center through pixel 'x' in image coordinates
# Returns ray position and ray direction (unit vector)
def camera_ray(P, x):
    o = cam_position(P)
    ox = np.dot(npla.pinv(P), x)

    flip = False
    if (ox[3] < 0):
        # print("Camera ray negative")
        flip = True
    if (np.isclose(ox[3], 0)):
        # print("Camera ray divide by 0")
        ox[3] = 1.0

    ox = ox / ox[3]
    ox = ox[:3] - o
    ox = ox / npla.norm(ox)

    if (flip):
        ox = ox * -1
    return o, ox


# Modified version of cam_position() for Numba
# Taken from: https://mathoverflow.net/a/68488/130043
# This function is 13 times faster than cam_position() (6us vs. 80us)
@jit(nopython=True, nogil=True)
def cam_position_2(P):
    pos = npla.pinv(P[:,:3]).dot(P[:,-1]) * -1
    pos = pos.reshape((3,1))
    return pos


# Modified version of camera_ray() for Numba
# This function is 15 times faster than camera_ray() (13us vs. 200us)
@jit(nopython=True, nogil=True)
def camera_ray_2(P, x):
    o = cam_position_2(P)
    x = x.astype(np.float64)
    ox = np.dot(npla.pinv(P), x)

    flip = False
    if (ox[3,0] < 0):
        # print("Camera ray negative")
        flip = True

    ox = ox / ox[3]
    ox = ox[:3] - o
    ox = ox / npla.norm(ox)

    if (flip):
        ox = ox * -1
    return o, ox


# Angle between two vectors in radians
# Taken from https://stackoverflow.com/a/13849249/7046003
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2':
    """
    v1 = v1 / npla.norm(v1)
    v2 = v2 / npla.norm(v2)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


# Compute camera position, orientation, and angle between all cameras
def cam_layout(P):
    ncam = len(P)
    POS = cam_position_all(P)
    ORN = cam_orientation_all(P)

    ANG = np.zeros((ncam, ncam))
    for i in range(ncam):
        for j in range(ncam):
            v1 = ORN[i].reshape((3))
            v2 = ORN[j].reshape((3))
            angle = angle_between(v1, v2)
            angle = np.rad2deg(angle)
            ANG[i,j] = angle
    return POS, ORN, ANG


# Project 3D point X in homogeneous coordinates to image plane
# Returns 2D point in homogeneous coordinates (x, y, 1)
@jit(nopython=True, nogil=True)
def cam_project(P, X):
    x = np.dot(P, X)
    x = x / x[2]
    return x


# Compute ZNCC similarity between two image patches.
# Returns a single scalar value representing the similarity between the two patches.
# ZNCC is in the range [-1, 1], where 1 means perfect positive correlation, -1 means
# perfect negative correlation, and 0 means no correlation.
# If the input patches have different sizes or are empty, this function returns -1.
@jit(nopython=True, nogil=True)
def zncc(im1, im2):
    r1, c1 = im1.shape
    r2, c2 = im2.shape

    if (r1==r2) and (c1==c2) and (r1>0) and (c1>0):
        mean1 = np.mean(im1)
        mean2 = np.mean(im2)
        std1 = np.std(im1)
        std2 = np.std(im2)
        # If either std is 0 the denominator becomes 0 and this returns NaN;
        # downstream callers are expected to handle NaNs explicitly.
        result = (im1-mean1) * (im2-mean2) / (std1 * std2)
        return np.round(np.sum(result)/(r1*c1), 4)
    else:
        return -1


# Python implementation of np.gradient since Numba (0.45.1) doesn't yet support np.gradient
# Returns gradients in x and y directions, and the gradient magnitude.
@jit(nopython=True, nogil=True)
def gradient(im):
    r, c = im.shape
    im = np.asarray(im, dtype=np.float32)
    gx = np.zeros((r, c))
    gy = np.zeros((r, c))

    for i in range(r):
        gx[i,0] = im[i, 1] - im[i, 0]           # First column
        gx[i,c-1] = im[i, c-1] - im[i, c-2]     # Last column

        for j in range(1, c-1):                 # Except first and last columns
            gx[i,j] = (im[i, j+1] - im[i, j-1]) / 2

    for j in range(c):
        gy[0,j] = im[1, j] - im[0, j]
        gy[r-1,j] = im[r-1, j] - im[r-2, j]

        for i in range(1, r-1):
            gy[i,j] = (im[i+1, j] - im[i-1, j]) / 2

    gm = (gx**2 + gy**2) ** 0.5

    return gx, gy, gm                           # np.gradient returns gy, gx


# Python implementation of np.clip since Numba (0.45.1) doesn't yet support np.clip
@jit(nopython=True, nogil=True)
def clip(a, low, high):
    return np.maximum(low, np.minimum(a, high))


# Compute distance map from depth map and edge map
def dist_map_transform(dmap, emap):
    from scipy import ndimage as ndi
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max

    support_size = 25
    fp = np.ones((support_size, support_size))
    local_maxi = peak_local_max(dmap, indices=False, footprint=fp, labels=emap)
    markers = ndi.label(local_maxi)[0]
    labels = watershed(dmap *-1, markers, mask=emap)
    n_regions = np.unique(markers).size

    out = np.zeros_like(labels)
    dmap_final = dmap.copy()

    for i in range(n_regions):
        idx = (labels == i)
        max_val = np.max(dmap[idx])
        out[idx] = max_val
        dmap_final[idx] = dmap_final[idx] + max_val / 3
    return dmap_final


# Gradient map-based adaptive median filter
# No @jit since Numba (0.45.1) doesn't support np.pad
def adaptive_median_gm(im, gm):
    r, c = np.shape(im)
    im_med = np.zeros((r,c))
    median_sizes = np.zeros((r,c))

    pad_max = int(np.max(gm) / 2)
    im_pad = np.pad(im, pad_max, mode='reflect')
    im_pad = np.float32(im_pad)

    for i in range(r):
        for j in range(c):
            pad_size = int(gm[i,j] / 2)

            icentre = i + pad_max
            jcentre = j + pad_max

            istart = icentre - pad_size
            iend = icentre + pad_size + 1
            jstart = jcentre - pad_size
            jend = jcentre + pad_size + 1

            patch = im_pad[istart:iend, jstart:jend]
            median_sizes[i,j] = patch.shape[0]

            assert(patch.shape[0] == patch.shape[1]), "Patch row != col"
            assert(patch.shape[0] % 2 == 1), "Patch size not odd number"
            assert(np.abs((np.round(gm[i,j]) - patch.shape[0])) <= 1), \
                "Patch-size {} doesnt match Median filter-size {}".format(gm[i,j], patch.shape[0])

            im_med[i,j] = np.nanmedian(patch)

    return im_med, median_sizes


# Gradient map-based adaptive Gaussian filter
# No @jit since Numba (0.45.1) doesn't support np.pad
import scipy.ndimage as ndimage
def adaptive_gaussian_gm(im, gm):
    r, c = np.shape(im)
    im_gauss = np.zeros((r,c))
    gauss_sizes = np.zeros((r,c))

    pad_max = int(np.max(gm) / 2)                           # Implicit floor
    im_pad = np.pad(im, pad_max, mode='reflect')

    for i in range(r):
        for j in range(c):
            pad_size = int(gm[i,j] / 2)

            i_centre = i + pad_max
            j_centre = j + pad_max

            i_start = i_centre - pad_size
            i_end = i_centre + pad_size + 1
            j_start = j_centre - pad_size
            j_end = j_centre + pad_size + 1

            patch = im_pad[i_start:i_end, j_start:j_end]
            gauss_sizes[i,j] = patch.shape[0]

            assert(patch.shape[0] == patch.shape[1]), "Patch row != col"
            assert(patch.shape[0] % 2 == 1), "Patch size not odd number"
            assert((np.round(gauss_sizes[i,j]) - patch.shape[0]) <= 1), \
                "Patch size doesnt match Gaussian filter size"

            gauss_sigma = (patch.shape[0] - 1) / (2 * 4)
            p_centre = int(patch.shape[0] / 2)
            im_gauss[i,j] = ndimage.gaussian_filter(patch, sigma=gauss_sigma)[p_centre, p_centre]

    return im_gauss, gauss_sizes


# Modified version of https://stackoverflow.com/a/29677616/7046003
# !!!: NaN in `values` is fine but in `weights` returns NaN
# !!!: Don't know fully how NaN values are handled (np.sort places them last)
@jit(nopython=True, nogil=True)
def weighted_quantile(values, weights, quantiles):
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'Quantiles should be in [0, 1]'

    v = values.flatten()
    w = weights.flatten()
    assert(len(v) == len(w)), "Values Weights size mismatch"

    sorter = np.argsort(v)
    v = v[sorter]
    w = w[sorter]

    weighted_quantiles = np.cumsum(w) - 0.5 * w
    weighted_quantiles /= np.sum(w)

    return np.interp(quantiles, weighted_quantiles, v)


# Fixed size weighted median filter applied to each pixel in an image
# !!!: Check weight normalization step
def im_weighted_median(im, w, s):
    r, c = np.shape(im)
    im_wmed = np.zeros((r,c))

    pad_size = int(s / 2)                                # Implicit floor
    im_pad = np.pad(im, pad_size, mode='reflect')
    w_pad = np.pad(w, pad_size, mode='reflect')

    for i in range(r):
        for j in range(c):
            i_end = i + 2 * pad_size + 1
            j_end = j + 2 * pad_size + 1

            im_patch = im_pad[i:i_end, j:j_end]
            w_patch = w_pad[i:i_end, j:j_end]

            # Normalize weights in a patch (?)
            w_patch = (w_patch - np.min(w_patch)) / (np.max(w_patch) - np.min(w_patch))

            assert(im_patch.shape[0] == im_patch.shape[1] == s), "Patch size mismatch"
            im_wmed[i,j] = weighted_quantile(im_patch, w_patch, np.array([0.5]))[0]

    return im_wmed


def mview2standard(fpath):
    fhandle = open(fpath, 'r')
    flines = fhandle.readlines()
    flines = [fline.rstrip() for fline in flines]
    ncam = int(flines[0])

    P = np.zeros((ncam,3,4))
    for i in range(ncam):
        fline = flines[i+1]
        params = fline.split()
        params = np.asarray(params[1:], dtype=np.float64)
        K = params[0:9].T.reshape((3,3))
        R = params[9:18].T.reshape((3,3))
        T = params[18:21].T.reshape((3,1))
        Pi = np.dot(K, np.hstack((R, T)))
        P[i] = Pi.copy()
    return P


# Load MVE files (depth maps and normal maps) and convert to standard format
def mvei2standard(file_path):
    with open(file_path, 'rb') as f:
        f.seek(11)                                              # Skip header (11 bytes)
        dims = np.fromfile(f, dtype=np.int32, count=4)          # width, height, channels, type
        depth_map = np.fromfile(f, dtype=np.float32, count=-1)

    depth_map = depth_map.reshape((dims[1], dims[0]))
    return depth_map


# Convert camera matrices from EPFL CVLab format to standard format used in this codebase
# https://documents.epfl.ch/groups/c/cv/cvlab-unit/www/data/multiview/denseMVS.html
def cvlab2standard(folder):
    files = os.listdir(folder)
    files = [f for f in files if f.lower().endswith('camera')]
    ncam = len(files)

    P = np.zeros((ncam,3,4))
    for i,file in enumerate(files):
        fpath = os.path.join(folder, file)
        fhandle = open(fpath, 'r')
        flines = fhandle.readlines()
        flines = [fline.rstrip('\n') for fline in flines]

        fline = flines[0] + flines[1] + flines[2]
        params = fline.split()
        K = np.asarray(params, dtype=np.float64).reshape((3,3))
        fline = flines[4] + flines[5] + flines[6]
        params = fline.split()
        R = np.asarray(params, dtype=np.float64).reshape((3,3))
        fline = flines[7]
        params = fline.split()
        T = np.asarray(params, dtype=np.float64).reshape((3,1))

        RT = np.hstack((R.T, np.dot(-R.T, T)))
        P[i] = np.dot(K, RT)
    return P


# Taken from https://stackoverflow.com/a/37882746/7046003
def interp_missing(a, method):
    from scipy.interpolate import griddata
    x, y = np.indices(a.shape)
    interp = np.array(a)
    mask = np.isnan(a)
    interp[np.isnan(interp)] = griddata(
                                (x[~mask], y[~mask]),           # points we know
                                a[~mask],                       # values we know
                                (x[mask], y[mask]),             # points to interpolate
                                method = method)                # nearest/linear/cubic
    return interp


# Cross-view consistency check for depth maps
@jit(nopython=True, nogil=True)
def crossview_consistency(imx0, D, P, ANG, ang_low, ang_high, k, d_tol):
    D_consist = D[imx0].copy()
    n_cam, n_row, n_col = D.shape
    X = np.ones((4,1))
    C = np.zeros_like(D[imx0])

    cam_idx = np.where((ANG[imx0] > ang_low) & (ANG[imx0] < ang_high))[0]

    for x in range(n_col):
        for y in range(n_row):
            t = D[imx0, y, x]

            if (t != 0):
                point = np.array((x,y,1)).reshape((3,1))
                rpos, rdir = camera_ray_2(P[imx0], point)
                X[0:3] = rpos + t * rdir

                for i,iv in enumerate(cam_idx):
                    pt = cam_project(P[iv], X)
                    xi = int(np.round(pt[0,0]))
                    yi = int(np.round(pt[1,0]))

                    rpos2, rdir2 = camera_ray_2(P[iv], pt)
                    t2 = (rdir2.T).dot(X[0:3] - rpos2)[0,0]

                    if (xi<0) or (yi<0) or (xi>=n_col) or (yi>=n_row):
                        pass

                    # Absolute threshold
                    elif (np.abs(D[iv,yi,xi] - t2) <= d_tol):
                        C[y,x] = C[y,x] + 1

    # Filter out depths with low cross-view consistency
    if (k > 1):                     # Fixed k neighbors
        idx = (C < k)
    else:                           # Percentage of neighbors
        n_neigh = len(cam_idx)
        k_neigh = int(np.round(k * n_neigh))
        idx = (C < k_neigh)

    for i in range(n_row):
        for j in range(n_col):
            if (idx[i,j]):
                D_consist[i,j] = 0

    return D_consist, C
