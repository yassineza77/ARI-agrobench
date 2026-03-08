#!/usr/bin/env python3
"""
optimize_and_validate_grid.py — Grid stress-evaluator over corruption families

• Loops over corruption families with predefined parameter grids.
• Can evaluate 1–5 models sequentially on the SAME corrupted set.
• One CSV PER FAMILY, with: params, P/R, mAP50, mAP50-95, inference time, FPS,
  parameters (#weights), GFLOPS, and GPU memory.
• When ≥2 models are selected, saves 10 sample images from the best-mAP50 dataset.

Usage:
  python3 optimize_and_validate_grid.py

Customize the CONFIG section below to select models, seeds, grids, etc.

Requirements:
  - ultralytics >= 8.x
  - prepare_valset.py in the same scripts/ directory
"""

import json, csv, time, os, sys, re, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

try:
    import torch
except Exception:
    torch = None  # CPU-only fallback

from ultralytics import YOLO

# ========================= CONFIG =========================
# Set these to match your data and model paths:

# Path to training data root
ROOT = Path.cwd()  # Change to your project root if needed

# Input: original validation set
VAL_ORI_IMG = ROOT / "data" / "val_ori" / "images"
VAL_ORI_LBL = ROOT / "data" / "val_ori" / "labels"

# Output: corrupted validation set
VAL_FIN_IMG = ROOT / "data" / "val_fin" / "images"
VAL_FIN_LBL = ROOT / "data" / "val_fin" / "labels"

# Path to data.yaml (YOLO format)
DATA_YAML_SRC = ROOT / "data" / "data.yaml"

# Inference settings
IMG_SIZE = 416
WORKERS_BATCH = {"workers": 0, "batch": 8}

# ============== MODELS (add your model paths) ==============
MODEL_PATHS = {
    # Example:
    # "yolov8n": "/path/to/yolov8n/weights/best.pt",
    # "custom_model": "/path/to/custom/model/best.pt",
}

# Select which models to evaluate (any subset of keys above)
SELECTED_MODELS = []  # e.g., ["yolov8n", "custom_model"]

# ========================= GRIDS =========================
# Global base knobs applied to ALL families
B_BRIGHT_GRID = [0.70, 1.00, 1.18]
B_BLUR_GRID   = [0, 2]
B_NOISE_GRID  = [0, 12.52]
HUE_FLAGS     = [False]  # True to include hue-jitter variants

# Family-specific corruption parameter grids
FAMILY_GRIDS = {
    "lowpass":   [{"cutoff": x} for x in [0.06, 0.08, 0.085, 0.10, 0.12]],
    "downup":    [{"scale": s} for s in [0.70, 0.50, 0.40, 0.30]],
    "motion":    [{"ksize": k, "angle": a} for k in [11, 21] for a in [-15, 0, 15]],
    "fog":       [{"beta": b, "A": A} for b in [0.05, 0.08, 0.11, 0.14] for A in [200, 220, 240]],
    "cutout":    [{"boxes": c, "max_frac": f} for c in [2, 4, 6, 8] for f in [0.10, 0.20, 0.30]],
    "vignette":  [{"strength": v} for v in [0.40, 0.70]],
    "posterize": [{"bits": n} for n in [5, 3]],
    "poisson":   [{"alpha": a, "sigma": s} for a in [0.005, 0.01, 0.02, 0.04] for s in [6, 10, 16]],
    "colorcast": [{"r": r, "g": g, "b": b} for (r, g, b) in [
        (1.5, 0.8, 0.6), (1.3, 1.0, 0.8), (1.1, 1.2, 0.8), (0.9, 1.3, 1.2)
    ]],
}

# Seeds for random reproductibility
SEEDS = [1004]

# ========================= MAIN EXECUTION =========================
RUN_NAME   = f"grid_eval_{int(time.time())}"
LOG_DIR    = ROOT / "outputs" / RUN_NAME
IMAGES_OUT = LOG_DIR / "best_samples"

# ========================= HELPERS =========================

def get_script_dir() -> Path:
    """Get directory of this script."""
    return Path(__file__).parent

def make_temp_yaml(val_img_dir: Path) -> Path:
    """Create a temporary data.yaml with the corrupted validation set path."""
    try:
        import yaml
    except ImportError:
        import sys
        print("[ERROR] PyYAML not found. Install with: pip install pyyaml")
        sys.exit(1)
    
    with open(DATA_YAML_SRC) as f:
        data = yaml.safe_load(f)
    
    data["val"] = str(val_img_dir)
    tmp = DATA_YAML_SRC.parent / f"tmp_val_{int(time.time()*1e6)}.yaml"
    
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f)
    
    return tmp


def call_prepare(family: str, params: dict, seed: int, hue: bool, clear: bool = True):
    """Invoke prepare_valset.py to generate corrupted validation set."""
    script_dir = get_script_dir()
    prep_script = script_dir / "prepare_valset.py"
    
    if not prep_script.exists():
        raise FileNotFoundError(f"prepare_valset.py not found at {prep_script}")
    
    cmd = [
        sys.executable, str(prep_script),
        "--input-images", str(VAL_ORI_IMG),
        "--input-labels", str(VAL_ORI_LBL),
        "--output-images", str(VAL_FIN_IMG),
        "--output-labels", str(VAL_FIN_LBL),
        "--family", family,
        "--params", json.dumps(params),
        "--seed", str(seed),
    ]
    if hue:
        cmd.append("--hue")
    if clear:
        cmd.append("--clear")
    
    import subprocess
    subprocess.run(cmd, check=True)


def try_compute_gflops_and_params(yolo_obj: YOLO, imgsz: int = 416) -> Tuple[Optional[float], Optional[int]]:
    """Estimate GFLOPS and parameter count (best-effort)."""
    params = None
    gflops = None
    try:
        mdl = yolo_obj.model
        params = sum(p.numel() for p in mdl.parameters())
    except Exception:
        pass

    # Try THOP if available
    if gflops is None:
        try:
            from thop import profile
            import torch
            mdl = yolo_obj.model.eval()
            dummy = torch.zeros(1, 3, imgsz, imgsz)
            flops, _ = profile(mdl, inputs=(dummy,), verbose=False)
            gflops = float(flops) / 1e9
        except Exception:
            pass
    
    return gflops, params


def run_val(yolo_obj: YOLO, data_yaml: Path) -> dict:
    """Run Ultralytics validation and return metrics."""
    if torch and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    res = yolo_obj.val(
        data=str(data_yaml),
        imgsz=IMG_SIZE,
        device=0 if (torch and torch.cuda.is_available()) else "cpu",
        workers=WORKERS_BATCH["workers"],
        batch=WORKERS_BATCH["batch"],
        verbose=False,
    )

    # Extract metrics
    P = float(np.array(res.box.p).item())
    R = float(np.array(res.box.r).item())
    mAP50 = float(np.array(res.box.map50).item())
    mAP5095 = float(np.array(res.box.map).item())

    # Inference time (ms per image)
    inf_ms = None
    try:
        inf_ms = float(res.speed.get("inference", None))
    except Exception:
        pass

    FPS = None
    if inf_ms and inf_ms > 0:
        FPS = 1000.0 / inf_ms

    # GPU memory
    gpu_mem_mb = None
    if torch and torch.cuda.is_available():
        try:
            gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        except Exception:
            pass

    return {
        "P": P, "R": R, "mAP50": mAP50, "mAP5095": mAP5095,
        "inf_ms": inf_ms, "FPS": FPS, "gpu_mem_mb": gpu_mem_mb,
    }


def ensure_csv(family: str) -> Path:
    """Create CSV header if file doesn't exist."""
    path = LOG_DIR / f"{family}.csv"
    if not path.exists():
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "ts", "family", "params_json", "hue", "seed",
                "model", "gflops", "params",
                "gpu_mem_mb", "P", "R", "mAP50", "mAP5095", "inf_ms", "FPS",
            ])
    return path


def copy_best_images(src_img_dir: Path, dst_dir: Path, max_imgs: int = 10):
    """Copy up to max_imgs sample images from src to dst."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    images = sorted([p for p in src_img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}])
    for i, p in enumerate(images[:max_imgs]):
        shutil.copy2(p, dst_dir / p.name)


@dataclass
class BestRecord:
    mAP50: float
    family: str
    params: dict
    hue: bool
    seed: int


# ========================= MAIN LOOP =========================

def main():
    # Validate paths
    if not VAL_ORI_IMG.exists() or not VAL_ORI_LBL.exists():
        print(f"[ERROR] Original validation set not found at {VAL_ORI_IMG} or {VAL_ORI_LBL}")
        print(f"Please set VAL_ORI_IMG and VAL_ORI_LBL correctly in this script.")
        return
    
    if not DATA_YAML_SRC.exists():
        print(f"[ERROR] data.yaml not found at {DATA_YAML_SRC}")
        return

    # Create output directories
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_OUT.mkdir(parents=True, exist_ok=True)

    # Load YOLO models
    yolos: Dict[str, YOLO] = {}
    selected = SELECTED_MODELS or []
    
    if not selected:
        print("[WARN] No models selected. Please configure SELECTED_MODELS in this script.")
        return
    
    for name in selected:
        mp = MODEL_PATHS.get(name)
        if not mp or not Path(mp).exists():
            print(f"[WARN] Skipping model {name}: path not found at {mp}")
            continue
        print(f"Loading model: {name} from {mp}")
        yolos[name] = YOLO(mp)

    if not yolos:
        print("[ERROR] No valid models loaded. Check SELECTED_MODELS and MODEL_PATHS.")
        return

    multi_model = (len(yolos) >= 2)

    # Precompute model metadata
    model_meta: Dict[str, Tuple[Optional[float], Optional[int]]] = {}
    for name, yo in yolos.items():
        gf, pn = try_compute_gflops_and_params(yo, IMG_SIZE)
        model_meta[name] = (gf, pn)

    # Track best record per model
    best_per_model: Dict[str, BestRecord] = {}

    # Main loop over families
    for family, grid in FAMILY_GRIDS.items():
        csv_path = ensure_csv(family)
        print(f"\n===== FAMILY: {family} (grid size {len(grid)}) =====")

        # Nested loops: hue × seed × base-knobs × family-params
        for hue in HUE_FLAGS:
            for seed in SEEDS:
                for b_bright in B_BRIGHT_GRID:
                    for b_blur in B_BLUR_GRID:
                        for b_noise in B_NOISE_GRID:
                            for p in grid:
                                params = dict(p)
                                # Merge base knobs
                                params.update({
                                    "b_bright": float(b_bright),
                                    "b_blur": int(b_blur),
                                    "b_noise": float(b_noise),
                                })

                                # Prepare dataset
                                try:
                                    call_prepare(family, params, seed=seed, hue=hue, clear=True)
                                except subprocess.CalledProcessError as e:
                                    print(f"[ERROR] Failed to prepare {family} dataset: {e}")
                                    continue

                                tmp_yaml = make_temp_yaml(VAL_FIN_IMG)
                                try:
                                    # Evaluate all models on the SAME set
                                    for name, yo in yolos.items():
                                        metrics = run_val(yo, tmp_yaml)
                                        gf, pn = model_meta.get(name, (None, None))

                                        # Log CSV row
                                        with open(csv_path, "a", newline="") as f:
                                            w = csv.writer(f)
                                            w.writerow([
                                                int(time.time()), family, json.dumps(params), int(hue), seed,
                                                name,
                                                (None if gf is None else round(gf, 3)),
                                                pn,
                                                (None if metrics["gpu_mem_mb"] is None else round(metrics["gpu_mem_mb"], 1)),
                                                round(metrics["P"], 3), round(metrics["R"], 3),
                                                round(metrics["mAP50"], 3), round(metrics["mAP5095"], 3),
                                                (None if metrics["inf_ms"] is None else round(metrics["inf_ms"], 2)),
                                                (None if metrics["FPS"] is None else round(metrics["FPS"], 2)),
                                            ])

                                        # Track best mAP50 per model
                                        br = best_per_model.get(name)
                                        if (br is None) or (metrics["mAP50"] > br.mAP50):
                                            best_per_model[name] = BestRecord(
                                                mAP50=metrics["mAP50"], family=family, 
                                                params=dict(params), hue=hue, seed=seed
                                            )
                                finally:
                                    tmp_yaml.unlink(missing_ok=True)

    # Save best sample images per model (if ≥2 models)
    if multi_model:
        print("\nSaving best-sample images per model…")
        for name, br in best_per_model.items():
            # Rebuild the best dataset to copy images
            try:
                call_prepare(br.family, br.params, seed=br.seed, hue=br.hue, clear=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Failed to recreate best dataset for {name}: {e}")
                continue
            
            # Copy 10 sample images
            out_dir = IMAGES_OUT / name
            copy_best_images(VAL_FIN_IMG, out_dir, max_imgs=10)
            
            # Write metadata
            meta = {
                "model": name,
                "family": br.family,
                "params": br.params,
                "hue": br.hue,
                "seed": br.seed,
                "mAP50": br.mAP50,
                "note": "10 sample images from the corrupted validation set that yielded best mAP50.",
            }
            with open(out_dir / "best_config.json", "w") as f:
                json.dump(meta, f, indent=2)

    print(f"\n✓ All done. Results in: {LOG_DIR}")


if __name__ == "__main__":
    main()
