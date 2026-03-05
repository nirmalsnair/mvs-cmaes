# mvs-cmaes

Reference implementation of the paper [Scalable Multi-View Stereo using CMA-ES and Distance Transform-based Depth Map Refinement](https://doi.org/10.1117/12.2587241) presented at the _International Conference on Machine Vision (ICMV)_, SPIE, 2021. 

## Overview

This project implements a dense Multi-View Stereo (MVS) reconstruction pipeline that estimates per-pixel depth maps from calibrated multi-view images and fuses them into a dense 3D model.

The method focuses on **scalability and robustness** in both textured and homogeneous regions, two common challenges in image-based 3D reconstruction. Unlike many existing MVS approaches that rely on spatial propagation or global regularization, this approach processes each pixel independently, enabling **pixel-level parallelization** and making the algorithm highly scalable for large datasets or multi-core systems.

The pipeline combines **photometric matching, evolutionary optimization, and adaptive depth filtering** to produce accurate and complete reconstructions.

## Key Ideas

### 1. Per-pixel depth estimation using evolutionary optimization

Depth for each pixel is estimated using Covariance Matrix Adaptation Evolution Strategy (CMA-ES), a derivative-free optimization algorithm that minimizes a photometric discrepancy function across multiple views.

### 2. Hybrid matching for textured and homogeneous regions

Different similarity measures are used depending on the local image structure:

* ZNCC template matching for **textured regions** to recover fine geometric details.
* DAISY feature descriptors for **homogeneous regions**, where template matching becomes unreliable.

A distance transform computed from image edges determines which similarity measure should be used at each pixel.

### 3. Distance transform-based adaptive depth refinement

Since depth is estimated independently per pixel, the initial depth maps can be noisy. To address this, the method applies a **distance transform-based adaptive median filter**:

* Small filters near edges to preserve geometric boundaries
* Larger filters in homogeneous regions to enforce smoothness

This refinement suppresses noise while preserving fine structural details.

### 4. Cross-view consistency filtering

Depth values are validated across neighboring views using a **cross-view consistency check** to remove outliers and ensure geometrically consistent depth maps.

### 5. 3D reconstruction

The final refined depth maps are:

1. Back-projected into 3D space
2. Merged into a dense point cloud
3. Converted into a watertight surface using Poisson surface reconstruction

## Key Features

* Pixel-level parallelization for high scalability
* Handles both textured and textureless regions
* Evolutionary optimization for robust depth estimation
* Adaptive depth smoothing based on image structure
* Produces dense and complete 3D reconstructions

## Evaluation

The method was evaluated on the **Middlebury Multi-View Stereo benchmark** datasets: TempleRing and DinoRing.

Results demonstrate:

* High reconstruction completeness (>98%)
* Competitive accuracy compared to state-of-the-art MVS methods

## Citation

```
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
