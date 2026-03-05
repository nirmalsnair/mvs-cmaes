# mvs-cmaes

Reference implementation of the paper [Scalable Multi-View Stereo using CMA-ES and Distance Transform-based Depth Map Refinement](https://doi.org/10.1117/12.2587241) presented at the _International Conference on Machine Vision (ICMV)_, SPIE, 2021. 

## Overview

This repository implements a **Multi-View Stereo (MVS)** pipeline that reconstructs a dense 3D model from calibrated multi-view images.

Per-pixel depths are estimated using **CMA-ES** optimization and refined using a distance transform–based adaptive filtering strategy. To handle different surface characteristics, the method combines **ZNCC** template matching for textured regions and **DAISY** descriptors for homogeneous regions. The resulting depth maps are filtered, validated across views, and fused into a dense 3D reconstruction.

### Key Features

* Pixel-level parallelization for high scalability
* Handles both textured and textureless regions
* Evolutionary optimization for robust depth estimation
* Adaptive depth smoothing based on image structure
* Produces dense and complete 3D reconstructions

## Method

The pipeline consists of the following steps:

1. **Depth estimation** using CMA-ES, a derivative-free optimization algorithm, to minimize photometric discrepancy across views.
2. **Hybrid matching strategy**
   * ZNCC template matching for textured regions
   * DAISY descriptors for homogeneous regions
   * A distance transform from image edges determines which similarity measure to use per pixel.
3. **Adaptive median filtering** based on the distance transform to suppress noise while preserving edges.
4. **Cross-view consistency filtering** to remove inconsistent depth estimates.
5. **Depth fusion and surface reconstruction** to produce a dense point cloud and mesh using **Poisson Surface Reconstruction**.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/nirmalsnair/mvs-cmaes.git
cd mvs-cmaes
```

### 2. Download a dataset

Download a calibrated multi-view dataset such as [Middlebury MVS](https://vision.middlebury.edu/mview/data/) (e.g., *DinoRing* or *TempleRing*):

Place the images and camera parameters in the dataset directory expected by the scripts.

### 3. Generate depth maps

Compute depth maps using the two matching strategies:

```bash
python dmap_zncc.py
python dmap_daisy.py
```

### 4. Merge depth maps

Merge the two depth map sets and export them to an [MVE scene](https://github.com/simonfuhrmann/mve/wiki/MVE-File-Format):

```bash
python merge_dmaps.py
```

This step combines the depth maps, applies filtering and consistency checks, and writes the results in **MVE scene format**.

### 5. Reconstruct the 3D model

The generated scene is compatible with **Multi-View Environment (MVE)**:

[https://github.com/simonfuhrmann/mve](https://github.com/simonfuhrmann/mve)

Use the MVE tools to convert depth maps into a point cloud and run **Poisson Surface Reconstruction** to obtain the final mesh.

## Citation

```bibtex
@inproceedings{nair2021scalable,
  title={Scalable multi-view stereo using CMA-ES and distance transform-based depth map refinement},
  author={Nair, Nirmal S and Nair, Madhu S},
  booktitle={Thirteenth International Conference on Machine Vision},
  volume={11605},
  pages={201--208},
  year={2021},
  organization={SPIE}
}
```
