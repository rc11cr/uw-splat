# UW-Splat: Physics-Constrained Underwater 3D Gaussian Splatting

## Abstract

Underwater scenes are challenging visual environments, as light propagation through water introduces range- and wavelength-dependent attenuation and scattering that severely distort appearance. Fully volumetric NeRF-based methods can jointly model both scene geometry and the participating medium, but they are slow to train and far from real-time at test time. In contrast, recent 3D Gaussian Splatting (3DGS) approaches enable rapid training and real-time rendering of clear-air scenes, yet they model only explicit geometry and surface radiance and therefore cannot account for the water column.

We introduce **UW-Splat**, a hybrid radiance-field representation that brings physically based underwater rendering to 3DGS. Our method represents scene geometry and clean object colors with 3D Gaussians and augments them with lightweight per-pixel MLPs that predict medium parameters—distance-dependent attenuation coefficients $(\sigma_d, \sigma_s)$ and asymptotic backscatter $B_\infty$—under a standard underwater image-formation model. By constraining 3DGS with this image-formation model and adding depth, opacity, and color regularizers, UW-Splat learns to decouple solid geometry from the water medium using only underwater image sequences. On the SeaThru-NeRF underwater dataset, UW-Splat outperforms state-of-the-art NeRF-based methods in reconstruction quality while retaining real-time rendering, substantially reducing the computational cost of high-quality underwater view synthesis.

---

## Setup

The below presumes **Ubuntu 22.04** with **CUDA 12.2** installed, but other versions of CUDA should be ok (the Docker container used in our experiments is **CUDA 11.8**).

### Clone the repo

```bash
git clone git@github.com:rc11cr/uw-splat.git --recursive
cd uw-splat
```
### Create a virtual environment

```bash
# create a virtual environment
conda create --name uwsplat_py310 -y python=3.10
conda activate uwsplat_py310
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```
## Data

**SeaThru-NeRF dataset** 
To keep consistency across different models, we recomputed the camera intrinsic/extrinsic parameters and performed distortion corrections using COLMAP's `image_undistorter` on the SeaThru-NeRF dataset:

```bash
colmap image_undistorter \
  --image_path /your_path_to_dataset/SeathruNeRF_dataset/IUI3-RedSea/images_wb \
  --input_path /your_path_to_dataset/SeathruNeRF_dataset/IUI3-RedSea/colmap/sparse/0 \
  --output_path /your_path_to_dataset/undistorted_seathrunerf_dataset/IUI3-RedSea \
  --output_type COLMAP
```
## Train a model

The training script builds on the original INRIA 3D Gaussian Splatting code, and **always** trains both:

* the 3DGS scene (geometry + clean color), and
* the hybrid water MLP (`HybridWaterNet`) with all underwater losses (DCP, gray-world, alpha background, saturation, etc.) configured in `WaterConfig` inside `train.py`.

`DATASET_PATH` must contain:

* an `images` (or undistorted images) folder, and
* a `sparse/0` COLMAP folder.

You can adjust defaults (iterations, learning rates, densification, etc.) in `arguments/__init__.py` and `WaterConfig` in `train.py`.

### Basic training

```bash
python train.py \
  -s /path/to/undistorted_seathrunerf_dataset/IUI3-RedSea \
  -m output/IUI3_UWSplat_run1 \
  --iterations 30000 \
  --eval \
  -r 1
```
**Notes:**

* `-s` / `--source_path` : dataset root (must contain `images` and `sparse/0`).
* `-m` / `--model_path`  : where checkpoints and renders are saved.
* `--iterations`       : total training steps (overrides the default `30_000`).
* `--eval`             : enables a fixed train/test split and test rendering.
* `-r` / `--resolution`: resolution scale (e.g., `1` = full res, `2` = half res).

All hybrid-water components are enabled inside `train.py` via:

* `HybridWaterNet(W=WaterConfig.MLP_WIDTH)`
* losses from `underwater/loss.py`:
    * `DarkChannelPriorLossV2`
    * `GrayWorldPriorLoss`
    * `SmoothDepthLoss`
    * `loss_backscatter`
    * `loss_spatial_consistency`
    * `loss_saturation`
    * `loss_opacity_entropy`
    * `loss_z_recon` (optional, depth-weighted L1)
    * `AlphaBackgroundLoss`

with their weights and schedules defined in the `WaterConfig` class (DCP annealing, depth-weighted recon, refinement stage, alpha priors, etc.).


