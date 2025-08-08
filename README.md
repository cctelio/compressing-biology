# Compressing biology
## Table of Contents
1. [Project overview](#project-overview)
2. [Folder structure](#folder-structure)
3. [Key scripts and notebooks](#key-scripts-and-notebooks)
4. [Steps](#steps)

## Project overview

This repository contains the code and some resources used for experiments involving biological image compression and analysis related to the paper:

**"Compressing Biology: Evaluating the Stable Diffusion VAE for Phenotypic Drug Discovery"**

### Folder structure

- **`cpg0000-jump-pilot/`**

Contains metadata and scripts required to run experiments with the CPJUMP1 dataset. Includes scripts for downloading and preprocessing the images.

- **`lsun/`**

Includes `partition_names.txt` for parallelizing inference on the LSUN dataset.

- **`dataset/`**

Contains files for defining `Dataset` objects used in inference.

- **`models/`**

Holds definitions of all pre-trained models and their associated inference functions.

### Key scripts and notebooks

- **`inference_cpg0000.py`**

Inference of pre-trained models on the CPJUMP1 dataset.

- **`inference_lsun.py`**

Inference of pre-trained models on the LSUN dataset.

- **`notebook.sh`**

Sets up the environment for running Jupyter notebooks on a cluster.

- **`phenotypic_activity.ipynb`**

Computes the phenotypic activity (FR) on the CPJUMP1 dataset.

- **`plots.ipynb`**

Generates plots for evaluation metrics including MAE, SSIM, EMD, KLD, and FID.

- **`requirements.txt`**

List of all libraries installed in the virtual environment using `pip3 freeze`.

## Steps

1. **Download CPJUMP1 images**

Run `cpg0000-jump-pilot/download.sh` to download the dataset.

2. **Preprocess images**

Use `processing.sh` to preprocess the downloaded images.

3. **Download LSUN dataset**

Follow instructions or use provided scripts to obtain the LSUN dataset.

4. **Run inference**

Execute `inference_cpg0000.sh` and `inference_lsun.sh` to perform inference on the respective datasets.

5. **Analyze results**

- Use `phenotypic_activity.ipynb` to compute phenotypic activity.

- Use `plots.ipynb` to visualize evaluation metrics.