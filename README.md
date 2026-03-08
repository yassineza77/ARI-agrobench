# Agri Robustness Index (ARI)

A normalized robustness metric for object detectors under agriculture-calibrated corruptions.

## Overview

ARI_AgroBench provides tools to evaluate the robustness of agricultural object detection models against realistic corruption types, including lighting variations, motion blur, fog, noise, and other weather-induced distortions common in real-world farming scenarios.

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

- **lowpass**: Low-pass filtering (out-of-focus blur)
- **downup**: Down-upsampling (resolution degradation)  
- **motion**: Motion blur (camera shake)
- **fog**: Fog/atmospheric haze
- **cutout**: Random occlusion patches
- **vignette**: Edge darkening
- **posterize**: Color quantization
- **poisson**: Poisson noise (sensor shot noise)
- **colorcast**: Color temperature shifts

## Installation

```bash
pip install -r requirements.txt
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{ari2024,
  title={Agro-Robustness Index (ARI): A Normalized Robustness Metric for Object Detectors Under Agriculture Calibrated Corruptions},
  author={Zarrouk, Yassine and Bourhaleb, Mohammed and Rahmoun, Mohammed and Hamdaoui, Hajar and Hacham, Khalid},
  year={2024}
}
```

## License

This project is licensed under the CC-BY-4.0 License - see [LICENSE](LICENSE) for details.
