#!/bin/bash
# Example usage of prepare_valset.py

# Prepare a single corruption: lowpass with specific parameters

python scripts/prepare_valset.py \
  --input-images data/val_ori/images \
  --input-labels data/val_ori/labels \
  --output-images data/val_fin/images \
  --output-labels data/val_fin/labels \
  --family lowpass \
  --params '{"cutoff":0.085, "b_bright":0.7, "b_blur":0, "b_noise":12.52}' \
  --seed 1004 \
  --clear

echo "Corrupted validation set ready at data/val_fin/"
