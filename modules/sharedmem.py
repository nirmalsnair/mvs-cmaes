"""
Shared memory utilities for multi-processing in EC-based line search.

This module provides functions to initialize and access shared memory for image data
and descriptors, allowing multiple processes to read from the same memory without
unnecessary copying.

This is used in the `dmap_zncc.py` and `dmap_daisy.py` scripts to enable shared-memory
multi-processing for the EC-based line search implemented in `modules/ec.py`.
"""

import numpy as np

def init(shared_base, shared_shape):
    global shared_data
    shared_data = np.ctypeslib.as_array(shared_base.get_obj())
    shared_data = shared_data.reshape(shared_shape)
