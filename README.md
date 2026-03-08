# Agri Robustness Index (ARI)

[![DOI](https://zenodo.org/badge/1175872492.svg)](https://doi.org/10.5281/zenodo.18909108)

A normalized robustness retention metric for object detectors evaluated under **agriculture-calibrated corruptions** (AgroBench).  
This repository provides the **benchmark specification**, **corrupted validation set generator**, and **ARI computation tools**.

---

## What is ARI?

**Agro-Robustness Index (ARI)** summarizes how much of a detector’s clean performance is retained, on average, under a fixed set of agriculture-calibrated corruption families and severity levels.

- **Clean capability:** measured on the clean validation/test set (e.g., mAP@50)
- **Robustness retention:** measured under corrupted inputs
- **ARI:** average retention across corruption families

In addition to the overall ARI score, the framework reports **per-family retention** to identify dominant failure modes (e.g., chromatic shift or occlusion).

## Quick Start

### 1. Prepare Corrupted Validation Set

```bash
python scripts/prepare_valset.py \
  --input-images path/to/val_ori/images \
  --input-labels path/to/val_ori/labels \
  --output-images path/to/val_fin/images \
  --output-labels path/to/val_fin/labels \
  --family lowpass \
  --params '{"cutoff":0.085, "b_bright":0.7, "b_blur":0, "b_noise":12.52}' \
  --seed 1004
```

### 2. Run Grid-Based Validation

```bash
python scripts/optimize_and_validate_grid.py
```
Customize model paths and corruption grids in the CONFIG section of the script.

## Corruption Families

- `lowpass`      Frequency attenuation (low-pass filtering)
- `downup`       Down-up sampling (resolution degradation)
- `motion`       Motion blur (platform/camera shake)
- `fog`          Fog/haze (contrast attenuation)
- `jpeg`         JPEG compression artifacts
- `cutout`       Structured occlusion (vegetation occlusion)
- `vignette`     Lens vignette (edge darkening)
- `posterize`    Bit-depth reduction (quantization)
- `colorcast`    Color cast (white-balance/channel scaling)

## Installation

```bash
pip install -r requirements.txt
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{Zarrouk2026ARI_Code,
  author       = {Zarrouk, Yassine},
  title        = {Agri-Robustness Index (ARI) and AgroBench},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18909108},
  url          = {https://doi.org/10.5281/zenodo.18909108}
}
```
If you used the methodology/metric of this work, please cite the manuscript:

```bibtex
@article{Zarrouk2026ARI_Manuscript,
  author  = {Zarrouk, Yassine and Khallou, Abdelhak and Bourhaleb, Mohammed and Rahmoun, Mohammed and Hamdaoui, Hajar and Hacham, Khalid},
  title   = {Agro-Robustness Index (ARI): A Normalized Robustness Metric for Object Detectors Under Agriculture Calibrated Corruptions},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
