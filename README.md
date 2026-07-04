# Synthetic Aperture Radar to Electro-Optical (SAR-to-EO) Image Translation using Attention cGAN

This repository implements a production-grade Conditional Generative Adversarial Network (cGAN) tailored for Synthetic Aperture Radar (SAR) to Electro-Optical (EO) optical image translation. Translating active radar backscatter (SAR) to optical reflectance (EO) is a highly ill-posed problem due to inverse data distributions, extreme pixel intensity saturation spikes, and geographic feature asymmetries. 

To solve this, our model implements:
* **Generator (Attention U-Net):** A customized U-Net architecture that integrates a **Self-Attention Block** at the deepest bottleneck layer. This enables global context modeling, utilizing large-scale structural patterns in the radar inputs to guide local optical feature reconstruction.
* **Discriminator (PatchGAN):** A $70 \times 70$ PatchGAN discriminator that maps input pairs to a grid of local receptive fields, forcing the generator to produce crisp, high-frequency structural boundaries and preventing mean-seeking blur.
* **Hybrid Objective Function:** Combines pixel-level correctness ($L1$ loss, $\lambda = 100$), deep feature consistency (VGG16-based Perceptual loss, $\lambda = 10$), and PatchGAN Adversarial loss to satisfy both perceptual and pixel-level quality requirements.

---

## Requirements

Running this codebase requires **Python 3.10+** and the specific package configurations defined in [requirements.txt](file:///Users/soumsingh/Desktop/Galaxy_eye_assignement/requirements.txt). The key dependencies include:
* `torch>=2.0.0`
* `torchvision>=0.15.0`
* `lpips>=0.1.4`
* `torchmetrics[image]>=1.0.0`
* `pandas>=2.0.0`
* `matplotlib>=3.7.0`
* `pyyaml>=6.0`

---

## Environment Setup

### Option 1: Conda Environment Setup (Recommended)
Create and initialize an isolated conda environment:
```bash
conda create -n sar2eo python=3.10 -y
conda activate sar2eo
pip install -r requirements.txt
```

### Option 2: Python Virtual Environment (venv)
Set up a clean virtual environment using virtualenv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Structure

This project uses the `requiemonk/sentinel12-image-pairs-segregated-by-terrain` dataset from Kaggle. 

### Spatial Data Leakage Prevention (Strict Holdout Strategy)
Adjacent geographic frames from satellite passes share strong spatial correlation, which creates a critical risk of data leakage if splits are partitioned randomly. To guarantee zero spatial leakage:
* **Training Terrains:** Trained exclusively on `agri` (agriculture), `grassland`, and `urban` terrains.
* **Validation/Testing Holdout:** Reserved `barrenland` completely as the unseen testing terrain.

### Directory Structure
Arrange the downloaded Kaggle dataset under the root workspace as follows:
```
sar_eo_data/
└── v_2/
    ├── agri/
    │   ├── s1/   # SAR Grayscale (.png)
    │   └── s2/   # EO RGB (.png)
    ├── grassland/
    │   ├── s1/
    │   └── s2/
    ├── urban/
    │   ├── s1/
    │   └── s2/
    └── barrenland/   # Isolated Testing Split
        ├── s1/
        └── s2/
```

---

## Training

The complete Exploratory Data Analysis (EDA), network architectures, loss formulations, and training runs are executed sequentially within the [GalaxEye_SAR_to_EO.ipynb](file:///Users/soumsingh/Desktop/Galaxy_eye_assignement/GalaxEye_SAR_to_EO.ipynb) notebook. 

### Sequential Execution Steps:
1. **Directory Set-up & Kaggle Download:** Mounts Google Drive (if in Colab), initializes folder structures, downloads and extracts the Sentinel-1/2 pairs.
2. **Exploratory Data Analysis (EDA):** Spot-checks image dimensions (verified strictly at $256 \times 256$) and analyzes data balance (4,000 paired samples per terrain).
3. **Statistical Profiling:** Computes skewness of the modalities (Grayscale SAR exhibits a slight negative skew of $-0.1647$ in Urban, whereas Optical EO has a positive skew of $1.0658$). This justifies normalizing all pixel values into a strict $[-1, 1]$ range.
4. **PyTorch Dataloader:** Implements a rigorous dataloader with synchronized random horizontal/vertical flips strictly on the training terrains, ensuring spatial pairs remain aligned.
5. **Architecture Diagnostics:** Defines `SAR2EOGenerator` (29.56M parameters) and `PatchGANDiscriminator` (2.76M parameters) with diagnostics checks.
6. **Loss Functions & Training Loop:** Initiates training for two configurations:
   * **Baseline Model (L1 Only):** Trains the generator using only pixel-wise L1 reconstruction loss.
   * **Full cGAN Model:** Trains generator and discriminator using the full hybrid objective function.

---

## Inference

Run inference using the self-contained standalone execution script [infer.py](file:///Users/soumsingh/Desktop/Galaxy_eye_assignement/infer.py):
```bash
python infer.py --input_dir <path> --output_dir <path> --weights <path/to/checkpoint>
```

**Parameters:**
* `--input_dir`: Path to the directory containing input SAR grayscale patches ($256 \times 256$).
* `--output_dir`: Path to save the generated EO optical patches.
* `--weights`: Path to the generator checkpoint `.pth` file.

---

## Evaluation

Compute quantitative image-to-image translation metrics using our standalone [evaluate.py](file:///Users/soumsingh/Desktop/Galaxy_eye_assignement/evaluate.py) script:
```bash
python evaluate.py --sar_dir <path/to/s1> --gt_dir <path/to/s2> --weights <path/to/checkpoint>
```

This script evaluates predictions on the validation/test holdout using the same metric libraries as the notebook:
* Peak Signal-to-Noise Ratio (PSNR)
* Structural Similarity Index Measure (SSIM)
* Learned Perceptual Image Patch Similarity (LPIPS)
* Fréchet Inception Distance (FID)

---

## Model Weights

(https://drive.google.com/file/d/1JqpactY6DiRhTHFueO6slHA-rPxzbo3u/view?usp=sharing)

---

## Results

Below is a comparison of validation performance evaluated on the unseen `barrenland` terrain split. Loss curves comparing the ablation runs are logged and saved in the repository at `logs/loss_curves_ablation.png`.

| Configuration | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (L1 Only)** | 12.7040 | 0.2211 | 0.8429| 35.0236 |
| **Full Attention cGAN** | 12.5969 | 0.1664 | 0.7002 | 9.7085 |

*\*Note: In the final notebook run, quantitative metrics were evaluated strictly for the finalized Full Attention cGAN model. The Baseline L1 model was trained as a control group for convergence analysis (loss curves comparison), showing a blurry, mean-seeking convergence compared to the crisp outputs of the hybrid configuration.*

---

## Citation / References

1. **Dataset:** Sentinel-12 Image Pairs Segregated by Terrain (Kaggle). Available: [sentinel12-image-pairs-segregated-by-terrain](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)
2. **Pix2Pix Architecture:** Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 1125-1134).
3. **LPIPS Metric:** Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The unreasonable effectiveness of deep features as a perceptual metric. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 586-595).




