
* ✅ Quick Start
* ✅ Full installation instructions
* ✅ Training (UNet & pix2pix)
* ✅ Inference (single & batch)
* ✅ Evaluation & performance
* ✅ Bands vs Indices
* ✅ Correct paths for your repository
* ✅ Correct CLI usage


````markdown
# sentinel1to2

Sentinel1to2 is a full deep-learning pipeline for translating **Sentinel-1 SAR data into Sentinel-2 optical bands or vegetation indices (NDVI, GNDVI, NDWI, etc.)**.

It supports:

- ✅ Supervised UNet training  
- ✅ pix2pix GAN training (UNet + PatchGAN)  
- ✅ Patch-wise inference with overlap and Gaussian blending  
- ✅ Full-scene inference  
- ✅ Evaluation with MAE, PSNR, SSIM, R², SAM  
- ✅ Bands and Vegetation Indices targets  
- ✅ Job-based experiment tracking  

Main entry point:

```bash
sentinel1to2
````

Implemented in:

```text
src/sentinel1to2/__main__.py
```

Repository:

```bash
https://github.com/gabriel-rmrz/sentinel1to2.git
```

---

# 🚀 QUICK START

```bash
# 1) Clone
git clone https://github.com/gabriel-rmrz/sentinel1to2.git
cd sentinel1to2

# 2) Setup environment
source scripts/setup/setup_python_env.sh
source .venv/bin/activate

# 3) Edit config paths
nano configs/test_config.yaml

# 4) Train
sentinel1to2 -c configs/test_config.yaml -s training

# 5) Inference
sentinel1to2 -c configs/test_config.yaml -s inference

# 6) Evaluation + performance
sentinel1to2 -c configs/test_config.yaml -s evaluation
sentinel1to2 -c configs/test_config.yaml -s performance
```

---

# 1️⃣ CLONE THE REPOSITORY

```bash
git clone https://github.com/gabriel-rmrz/sentinel1to2.git
cd sentinel1to2
```

---

# 2️⃣ INSTALLATION & ENVIRONMENT

## ✅ Automatic installation (recommended)

```bash
source scripts/setup/setup_python_env.sh
source .venv/bin/activate
```

This will:

* create `.venv`
* upgrade `pip`
* install the project in editable mode:

```bash
pip install -e .
```

---

## ✅ Manual installation (alternative)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

Dependencies are defined in:

```text
pyproject.toml
```

Main libraries:

* torch, torchvision, segmentation-models-pytorch
* numpy, pandas, scikit-learn, scikit-image
* rasterio, tqdm, matplotlib, pyyaml

---

# 3️⃣ CONFIGURATION SYSTEM

Everything is controlled through YAML files in:

```text
configs/
```

Main example:

```text
configs/test_config.yaml
```

Key sections:

```yaml
job:
  dir: "jobs/job_test"

preprocessing:
  patch_dimension: [128, 128]
  input_dir: "/path/to/train/data"

model:
  name: "SMP_UNet"
  parameters:
    encoder_name: "resnet34"
    in_channels: 4
    out_channels: 9

target:
  type: "bands"   # or "indices"

training:
  epochs: 100
  patience: 5
  learning_rate: 1e-4
  model_output: "smp_unet_best.pth"

training:
  gan:
    mode: "none"      # set to "pix2pix" to enable GAN
    lambda_recon: 100
    lambda_adv: 1.0

inference:
  input_dir: "/path/to/test/data"
  output_dir: "output_combined"

performance:
  bands_metric_names: ["mae", "psnr", "ssim"]
  indices_metric_names: ["mae", "psnr", "ssim", "r2", "sam"]
```

---

# 4️⃣ TRAINING

## ✅ 4.1 Supervised UNet

```yaml
training:
  gan:
    mode: "none"
```

Run:

```bash
sentinel1to2 -c configs/test_config.yaml -s training
```

---

## ✅ 4.2 pix2pix GAN Training

```yaml
training:
  gan:
    mode: "pix2pix"
    lambda_recon: 100.0
    lambda_adv: 1.0
```

Run:

```bash
sentinel1to2 -c configs/test_config.yaml -s training
```

---

# 5️⃣ LOSS SELECTION

## 🔵 BANDS (Sentinel-2)

```yaml
target:
  type: "bands"

training:
  loss:
    name: "CombinedLoss"
    parameters:
      alpha: 1.0
      beta: 2.0
      gamma: 0.1
      rgb_indices: [6, 2, 1]
```

Loss used:

```
L = α·L1 + β·SAM + γ·VGG
```

---

## 🟢 INDICES (NDVI, etc.)

```yaml
target:
  type: "indices"

training:
  loss:
    name: "IndexStructureLoss"
    parameters:
      alpha: 1.0     # L1
      beta: 1.0      # multi-scale gradients
      gamma: 0.2     # SSIM
      num_scales: 3
      window_size: 11
      sigma: 1.5
```

Loss used:

```
L = L1 + Gradients + SSIM
```

✅ No VGG
✅ No SAM
✅ Physically meaningful for vegetation indices

---

# 6️⃣ INFERENCE (FULL SCENE, PATCH-WISE)

```bash
sentinel1to2 -c configs/test_config.yaml -s inference
```

Features:

* Sliding window patching
* Overlapping patches
* Gaussian blending
* Automatic edge handling

Results saved in:

```text
<job.dir>/data/<inference.output_dir>/
```

---

# 7️⃣ BATCH INFERENCE

```bash
python batch_run_inference.py \
  -c configs/test_config.yaml \
  -i /path/to/test/scenes \
  -o /path/to/output_dir
```

---

# 8️⃣ EVALUATION

```bash
sentinel1to2 -c configs/test_config.yaml -s evaluation
```

Metrics:

* MAE
* PSNR
* SSIM
* R²
* SAM

Results in:

```text
<job.dir>/outputs/tables/
```

---

# 9️⃣ PERFORMANCE & PLOTS

```bash
sentinel1to2 -c configs/test_config.yaml -s performance
```

Generates:

* Error histograms
* SSIM / R² / SAM distributions
* Scatter plots

Saved in:

```text
<job.dir>/outputs/plots/
```

---

# 🔟 JOB DIRECTORY STRUCTURE

```text
jobs/job_test/
  data/
    smp_unet_best.pth
    normalization_params.npz
    output_combined/
      test_scene_pred.tif
    lists/
      train_scenes_list.csv
      val_scenes_list.csv
      test_scenes_list.csv
  outputs/
    tables/
    plots/
```

---

# ✅ END-TO-END PIPELINE

```bash
sentinel1to2 -c configs/test_config.yaml -s training
sentinel1to2 -c configs/test_config.yaml -s inference
sentinel1to2 -c configs/test_config.yaml -s evaluation
sentinel1to2 -c configs/test_config.yaml -s performance
```

---

# ✅ GPU USAGE

```bash
CUDA_VISIBLE_DEVICES=0 sentinel1to2 -c configs/test_config.yaml -s training
```

---

# ✅ SUMMARY OF MODES

| Task              | Model           | Loss             | GAN |
| ----------------- | --------------- | ---------------- | --- |
| S1 → S2 Bands     | UNet            | L1 + SAM + VGG   | ❌   |
| S1 → NDVI         | UNet            | L1 + Grad + SSIM | ❌   |
| pix2pix S1 → S2   | UNet + PatchGAN | L1 + SAM + VGG   | ✅   |
| pix2pix S1 → NDVI | UNet + PatchGAN | L1 + Grad + SSIM | ✅   |

---

# ✅ CITATION

> Sentinel-1 to Sentinel-2 translation with multi-scale structural losses and pix2pix GAN supervision.

---

# ✅ CONTACT & CONTRIBUTIONS

Issues, pull requests and extensions are welcome.

```


### TODO
- write a wraper
  - Chose model
  - simple, complete (detailed) output
  - create a directory with all the files necessary to run again the job, outputs, logs and errors
- Set requirements.txt
- plotting directory
- make a directory every time is run
- save model trained
