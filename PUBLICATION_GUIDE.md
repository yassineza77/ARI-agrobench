# ARI AgroBench - Publication Checklist

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   # or
   conda env create -f environment.yml
   ```

2. **Configure paths** in `scripts/optimize_and_validate_grid.py`:
   - Set `ROOT` to your project directory
   - Set `VAL_ORI_IMG`, `VAL_ORI_LBL` paths to your original validation data
   - Set `DATA_YAML_SRC` to point to your data.yaml file
   - Add your model paths to `MODEL_PATHS` and select them in `SELECTED_MODELS`

3. **Prepare a single corrupted validation set:**
   ```bash
   bash examples/example_run.sh
   ```

4. **Run grid evaluation on all corruption families:**
   ```bash
   python scripts/optimize_and_validate_grid.py
   ```

## Project Structure

```
ARI_AgroBench/
├── README.md                           # Main documentation
├── CITATION.cff                        # Citation metadata
├── LICENSE                             # CC-BY-4.0 license
├── requirements.txt                    # Python dependencies
├── environment.yml                     # Conda environment
│
├── scripts/
│   ├── prepare_valset.py              # Generate corrupted datasets
│   └── optimize_and_validate_grid.py   # Grid-based evaluation
│
├── configs/
│   ├── corruptions.yaml               # Parameter descriptions
│   └── eval.yaml                      # Grid configuration example
│
└── examples/
    ├── example_run.sh                 # Minimal usage example
    └── README_FORMAT.md               # Data format documentation
```

## Key Files Explained

### `scripts/prepare_valset.py`
Generates a corrupted validation set from an original dataset.

**Corruption families:**
- lowpass, downup, motion, fog, jpeg, cutout, vignette, posterize, poisson, colorcast

**Usage:**
```bash
python scripts/prepare_valset.py \
  --input-images path/to/val_ori/images \
  --input-labels path/to/val_ori/labels \
  --output-images path/to/val_fin/images \
  --output-labels path/to/val_fin/labels \
  --family lowpass \
  --params '{"cutoff":0.085, "b_bright":0.7, "b_blur":0, "b_noise":12.52}' \
  --seed 1004 \
  --clear
```

### `scripts/optimize_and_validate_grid.py`
Stress-tests models on parameter grids across all corruption families.

**Output:** One CSV per family with:
- Model performance (P, R, mAP50, mAP50-95)
- Inference speed (FPS, ms/image)
- Hardware usage (GPU memory, GFLOPS)
- Corruption parameters

**Best-performing examples:** If ≥2 models, saves 10 sample images + metadata.

## Data Format

Data should be in YOLO format:
```
data/
├── val_ori/
│   ├── images/         # Original validation images
│   └── labels/         # YOLO bounding box annotations
└── data.yaml           # Dataset config (nc, names, paths)
```

Label format (YOLO):
```
<class_id> <x_center> <y_center> <width> <height>  # all normalized [0,1]
```

## Publication Notes

- **MIT-friendly setup:** All code is reproducible and configurable
- **No hardcoded paths:** Uses command-line arguments and config files
- **Modular scripts:** Can run prepare_valset.py independently or in batch
- **Output format:** CSV results are easy to parse and visualize
- **Sample images included:** Optional—useful for supplementary material

## Modification for Your Paper

Before publishing, you may want to:

1. **Update `CITATION.cff`:**
   - Add your actual arXiv/DOI
   - Update author names and affiliations

2. **Customize `configs/eval.yaml`:**
   - Adjust grid parameters to your research scope
   - Add/remove corruption families as needed

3. **Add example images to `examples/`:**
   - Include 2-3 sample inputs + outputs for each family
   - These help readers understand the corruptions

4. **Update paths in `scripts/optimize_and_validate_grid.py`:**
   - Comment out unused models
   - Set default paths if distributing with sample data

## Requirements

- Python 3.8+
- PyTorch (CPU or GPU)
- Ultralytics YOLO >= 8.0
- NumPy, Pillow, OpenCV, tqdm (see requirements.txt)

## License

CC-BY-4.0: You may use and adapt this code for research. Please cite the paper.

---

**Questions?** See examples/README_FORMAT.md for detailed data format documentation.
