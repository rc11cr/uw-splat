# UW-Splat: Physics-Constrained Underwater 3D Gaussian Splatting

## Abstract

Underwater scenes are challenging visual environments, as light propagation through water introduces range- and wavelength-dependent attenuation and scattering that severely distort appearance. Fully volumetric NeRF-based methods can jointly model both scene geometry and the participating medium, but they are slow to train and far from real-time at test time. In contrast, recent 3D Gaussian Splatting (3DGS) approaches enable rapid training and real-time rendering of clear-air scenes, yet they model only explicit geometry and surface radiance and therefore cannot account for the water column.  

We introduce **UW-Splat**, a hybrid radiance-field representation that brings physically based underwater rendering to 3DGS. Our method represents scene geometry and clean object colors with 3D Gaussians and augments them with lightweight per-pixel MLPs that predict medium parameters—distance-dependent attenuation coefficients \((\sigma_d, \sigma_s)\) and asymptotic backscatter \(B_\infty\)—under a standard underwater image-formation model. By constraining 3DGS with this image-formation model and adding depth, opacity, and color regularizers, UW-Splat learns to decouple solid geometry from the water medium using only underwater image sequences. On the SeaThru-NeRF underwater dataset, UW-Splat outperforms state-of-the-art NeRF-based methods in reconstruction quality while retaining real-time rendering, substantially reducing the computational cost of high-quality underwater view synthesis.

---

## Setup

The below presumes **Ubuntu 22.04** with **CUDA 12.2** installed, but other versions of CUDA should be ok  
(the Docker container used in our experiments is **CUDA 11.8**).

### Clone the repo

```bash
git clone git@github.com:rc11cr/uw-splat.git --recursive
cd uw-splat

