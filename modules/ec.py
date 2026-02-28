"""
Evolutionary computation based 1D line-search utilities for depth estimation.

This module provides evolutionary search routines used by the driver scripts
`dmap_zncc.py` and `dmap_daisy.py`:

- ZNCC-based search: `_ls_fitness()`, `line_search()`, `line_search_shm()`
- DAISY-based search: `_ls_daisy_fitness()`, `line_search_daisy()`, `line_search_daisy_shm()`

The high-level configuration (population size, number of generations, etc.) is controlled from the
calling scripts.
"""

import numpy as np
from numba import jit
import pygmo as pg
import modules.func as func


# ZNCC similarity fitness function for EC-based line search.
@jit(nopython=True, nogil=True)
def _ls_fitness(t, I, P, camlist, p0, rpos, rdir, psize, k_best):
    X = np.ones((4,1))
    X[0:3] = rpos + t * rdir
    nft = len(camlist)
    S = np.zeros((nft))

    for idx, val in enumerate(camlist):
        xi = func.cam_project(P[val], X)
        pi = func.get_patch(I[val], xi[0,0], xi[1,0], psize)
        S[idx] = func.zncc(p0, pi)

    if (k_best >= 1):
        k = k_best
        S = S[~np.isnan(S)]
        S = np.sort(S)[::-1]            # Sort values in descending order (ZNCC)
        S = S[0:k]                      # Consider only 'k' highest values
        f = np.nansum(S)
    else:
        k = np.floor(nft * k_best)
        S = S[~np.isnan(S)]
        S = np.sort(S)[::-1]            # Sort values in descending order (ZNCC)
        S = S[0:k]                      # Consider only 'k' highest values
        f = np.nansum(S)
    return [-f]                         # Higher ZNCC is better but pygmo minimizes by default, hence the negative sign


# EC-based line search that minimizes the ZNCC similarity fitness function defined above.
# The image data is accessed directly (no shared memory), for single-core execution.
def line_search(I, P, B, ang, idx, x, y, psize, npop, ngen, ANGBOUND, k_best):
    class Search(object):
        def fitness(self, t):
            return _ls_fitness(t, I, P, camlist, p0, rpos, rdir, psize, k_best)
        def get_bounds(self):
            return ([B[0]], [B[1]])     # Problem dim is inferred from bound dim

    camlist = np.where((ang > ANGBOUND[0]) & (ang < ANGBOUND[1]))[0]
    point = np.array((x,y,1)).reshape((3,1))
    rpos, rdir = func.camera_ray(P[idx], point)
    p0 = func.get_patch(I[idx], x, y, psize)

    algo = pg.algorithm(pg.cmaes(gen=ngen))
    prob = pg.problem(Search())
    pop = pg.population(prob=prob, size=npop)
    pop = algo.evolve(pop)

    return [pop.champion_x, pop.champion_f]


# EC-based line search with shared memory for multi-processing.
# The fitness function is the same as above, but the image data is accessed from
# shared memory to avoid unnecessary copying across processes.
def line_search_shm(P, B, ang, idx, x, y, psize, npop, ngen, ANGBOUND, k_best):
    import modules.sharedmem as sm
    I = sm.shared_data

    class Search(object):
        def fitness(self, t):
            return _ls_fitness(t, I, P, camlist, p0, rpos, rdir, psize, k_best)
        def get_bounds(self):
            return ([B[0]], [B[1]])     # Problem dim is inferred from bound dim

    camlist = np.where((ang > ANGBOUND[0]) & (ang < ANGBOUND[1]))[0]
    point = np.array((x,y,1)).reshape((3,1))
    rpos, rdir = func.camera_ray(P[idx], point)
    p0 = func.get_patch(I[idx], x, y, psize)

    algo = pg.algorithm(pg.pso(gen=ngen))
    prob = pg.problem(Search())
    pop = pg.population(prob=prob, size=npop)
    pop = algo.evolve(pop)

    return [pop.champion_x, pop.champion_f]


# DAISY similarity fitness function for EC-based line search.
@jit(nopython=True, nogil=True)
def _ls_daisy_fitness(t, D, P, cam_list, d0, ray_pos, ray_dir, k_best):
    X = np.ones((4,1))
    X[0:3] = ray_pos + t * ray_dir

    n_cam, n_rows, n_cols, _ = D.shape

    nft = len(cam_list)
    S = np.zeros((nft))

    for idx, val in enumerate(cam_list):
        xi = func.cam_project(P[val], X)
        x = int(np.round(xi[0,0]))
        y = int(np.round(xi[1,0]))

        if (x<0) or (y<0) or (x>=n_cols) or (y>=n_rows):
            S[idx] = 9999                       # A large value to discourage
        else:
            di = D[val, y, x]                   # (row,col) is (y,x) in image coordinates 
            S[idx] = np.linalg.norm(d0 - di)

    k = k_best
    S = np.sort(S)                              # Sort values in ascending order
    S = S[0:k]                                  # Consider only 'k' lowest values
    f = np.sum(S)

    return [f]


# EC-based line search that minimizes the DAISY similarity fitness function defined above.
# The image descriptors are accessed directly (no shared memory), for single-core execution.
def line_search_daisy(D, P, B, angles, idx, x, y, n_pop=20, n_gen=30, k_best=5):
    class Search(object):
        def fitness(self, t):
            return _ls_daisy_fitness(t, D, P, cam_list, d0, ray_pos, ray_dir, k_best)
        def get_bounds(self):
            return ([B[0]], [B[1]])             # Problem dim is inferred from bound dim

    ANGBOUND = [5, 60]                          # TODO: softcode
    cam_list = np.where((angles > ANGBOUND[0]) & (angles < ANGBOUND[1]))[0]

    point = np.array((x,y,1)).reshape((3,1))
    ray_pos, ray_dir = func.camera_ray(P[idx], point)

    d0 = D[idx, y, x]                           # !!!: (x,y) -> (r,c)

    algo = pg.algorithm(pg.cmaes(gen=n_gen))
    prob = pg.problem(Search())
    pop = pg.population(prob=prob, size=n_pop)
    pop = algo.evolve(pop)

    return [pop.champion_x, pop.champion_f]


# EC-based line search that minimizes the DAISY similarity fitness function defined above.
# The image descriptors are accessed from shared memory for multi-processing.
def line_search_daisy_shm(P, B, angles, idx, x, y, n_pop, n_gen, ANGBOUND, k_best):
    import modules.sharedmem as sm
    D = sm.shared_data

    class Search(object):
        def fitness(self, t):
            return _ls_daisy_fitness(t, D, P, cam_list, d0, ray_pos, ray_dir, k_best)
        def get_bounds(self):
            return ([B[0]], [B[1]])             # Problem dim is inferred from bound dim

    cam_list = np.where((angles > ANGBOUND[0]) & (angles < ANGBOUND[1]))[0]

    point = np.array((x,y,1)).reshape((3,1))
    ray_pos, ray_dir = func.camera_ray(P[idx], point)

    d0 = D[idx, y, x]                           # (row,col) is (y,x) in image coordinates

    algo = pg.algorithm(pg.cmaes(gen=n_gen))
    prob = pg.problem(Search())
    pop = pg.population(prob=prob, size=n_pop)
    pop = algo.evolve(pop)

    return [pop.champion_x, pop.champion_f]
