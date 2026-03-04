"""
Convert MVE image files (depth maps and normal maps) to standard format for visualization
and further processing.

The MVE image file format is a custom binary format used by the MVE library for storing
depth maps and normal maps.

The MVE image file format starts with the signature 89 4D 56 45 5F 49 4D 41 47 45 0A (hex).
  89: Has the high bit set to discriminate from text files.
  4D 56 45 5F 49 4D 41 47 45: The ASCII letters MVE_IMAGE.
  0A: A Unix newline character.

Refer: https://github.com/simonfuhrmann/mve/wiki/MVE-File-Format
"""

import numpy as np
import matplotlib.pyplot as plt


# Depth map
# fpath = 'data\\MVE-20190610\\scene1\\views\\view_0000.mve\\smvs-B0.mvei'

# Normal map
fpath = 'data\\scene-dinoRing-smvs\\views\\view_0000.mve\\smvs-B0N.mvei'


# Read MVE image file
# The first 11 bytes are the signature, followed by 4 int32 values for width, height, channels, and type,
# and then the rest of the file is the image data as float32 values.
with open(fpath, 'rb') as f:
    f.seek(11)                                            # Skip signature (11 bytes)
    dims = np.fromfile(f, dtype=np.int32, count=4)        # width, height, channels, type
    data = np.fromfile(f, dtype=np.float32, count=-1)     # image data

# Depth map
# dmap = data.reshape((dims[1], dims[0]))

# Normal map
dmap = data.reshape((dims[1], dims[0], 3))

# Replace background with NaN
d2 = dmap.copy()
d2[d2==0] = np.nan

# # Plot the depth map
# vmin = np.nanmin(d2)
# vmax = np.nanmax(d2)
# plt.imshow(d2, vmin=vmin, vmax=vmax)

# Plot the normal map channels separately
fig, axes = plt.subplots(nrows=1, ncols=3)
axes[0].imshow(d2[:,:,0])
axes[1].imshow(d2[:,:,1])
axes[2].imshow(d2[:,:,2])
[axi.axis('off') for axi in axes]
fig.tight_layout()
