#!/usr/bin/env python3
"""
prepare_valset.py — build a corrupted validation set under val_fin/ based on
a chosen manipulation 'family' and parameters (passed as JSON).

Example:
python3 prepare_valset.py \
  --input-images /path/to/val_ori/images \
  --input-labels /path/to/val_ori/labels \
  --output-images /path/to/val_fin/images \
  --output-labels /path/to/val_fin/labels \
  --family lowpass \
  --hue \
  --params '{"cutoff":0.08, "b_bright":1.05, "b_blur":2, "b_noise":6.0}' \
  --seed 123 --clear
"""
import argparse, json, shutil, random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from tqdm import tqdm

# ---------------- base ops ----------------
def add_gaussian_noise(arr, mean=0.0, std=10.0):
    noise = np.random.normal(mean, std, size=arr.shape).astype(np.float32)
    out   = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

def jitter_hue(img: Image.Image, max_deg=30):
    hsv = img.convert("HSV")
    arr = np.array(hsv, dtype=np.uint16)
    shift = int((random.uniform(-max_deg, max_deg) / 360.0) * 255.0)
    arr[..., 0] = (arr[..., 0].astype(np.int32) + shift) % 256  # fix overflow
    return Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")

def down_up(img: Image.Image, scale=0.6, resample=Image.BILINEAR):
    w, h = img.size
    small = img.resize((max(1, int(w*scale)), max(1, int(h*scale))), resample)
    return small.resize((w, h), resample)

from io import BytesIO
def jpeg_compress(img: Image.Image, quality=30):
    buf = BytesIO(); img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0); return Image.open(buf).convert("RGB")

def motion_blur(img: Image.Image, ksize=9, angle=15):
    ksize = int(ksize) | 1
    k = np.zeros((ksize, ksize), np.float32)
    k[ksize//2, :] = 1.0
    M = cv2.getRotationMatrix2D((ksize/2, ksize/2), angle, 1)
    k = cv2.warpAffine(k, M, (ksize, ksize))
    s = k.sum();  k = k / (s if s != 0 else 1.0)
    arr = cv2.filter2D(np.array(img), -1, k)
    return Image.fromarray(arr)

def poisson_gaussian(arr: np.ndarray, alpha=0.01, sigma=8.0):
    lam = np.clip(arr.astype(np.float32) * alpha, 0, 255)
    shot = np.random.poisson(lam).astype(np.float32) / max(alpha, 1e-6)
    read = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    out = np.clip(shot + read, 0, 255).astype(np.uint8)
    return out

def add_fog(img: Image.Image, beta=0.08, A=220):
    arr = np.array(img).astype(np.float32)
    t = np.exp(-beta)
    out = arr * t + A * (1 - t)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def vignette(img: Image.Image, strength=0.6):
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h/2.0, w/2.0
    r = np.sqrt((y - cy)**2 + (x - cx)**2) / np.sqrt(cx**2 + cy**2)
    mask = 1.0 - strength * (r**2)
    mask = np.clip(mask, 0.4, 1.0)[..., None]
    return Image.fromarray(np.clip(arr * mask, 0, 255).astype(np.uint8))

def cutout(img: Image.Image, boxes=5, max_frac=0.2):
    arr = np.array(img)
    h, w = arr.shape[:2]
    for _ in range(int(boxes)):
        fh, fw = np.random.uniform(0.05, max_frac, 2)
        ch, cw = max(1, int(h*fh)), max(1, int(w*fw))
        y = np.random.randint(0, max(1, h - ch)); x = np.random.randint(0, max(1, w - cw))
        arr[y:y+ch, x:x+cw] = np.random.randint(0, 255, (ch, cw, 3), dtype=np.uint8)
    return Image.fromarray(arr)

def fft_lowpass(img: Image.Image, cutoff=0.08):
    arr = np.array(img).astype(np.float32)
    out = []
    for c in range(3):
        F = np.fft.fftshift(np.fft.fft2(arr[..., c]))
        h, w = F.shape; cy, cx = h//2, w//2
        R = int(max(1, min(h, w) * cutoff))
        mask = np.zeros_like(F); mask[cy-R:cy+R, cx-R:cx+R] = 1
        G = F * mask
        out.append(np.real(np.fft.ifft2(np.fft.ifftshift(G))))
    out = np.stack(out, -1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

def color_cast(img: Image.Image, r=1.0, g=0.9, b=0.8):
    arr = np.array(img).astype(np.float32)
    M = np.array([r, g, b], dtype=np.float32).reshape(1, 1, 3)
    out = np.clip(arr * M, 0, 255).astype(np.uint8)
    return Image.fromarray(out)

def posterize(arr: np.ndarray, bits=4):
    bits = int(bits)
    levels = 2 ** bits
    step = max(1, 256 // levels)
    return np.clip((arr // step) * step, 0, 255).astype(np.uint8)

# ---------------- families ----------------
def apply_family(img: Image.Image, family: str, p: dict) -> Image.Image:
    if family == "lowpass":     return fft_lowpass(img, cutoff=p["cutoff"])
    if family == "downup":      return down_up(img, scale=p["scale"])
    if family == "motion":      return motion_blur(img, ksize=p["ksize"], angle=p["angle"])
    if family == "fog":         return add_fog(img, beta=p["beta"], A=p.get("A", 220))
    if family == "jpeg":        return jpeg_compress(img, quality=p["quality"])
    if family == "cutout":      return cutout(img, boxes=p["boxes"], max_frac=p["max_frac"])
    if family == "vignette":    return vignette(img, strength=p["strength"])
    if family == "posterize":   return Image.fromarray(posterize(np.array(img), bits=p["bits"]))
    if family == "poisson":     return Image.fromarray(poisson_gaussian(np.array(img), alpha=p["alpha"], sigma=p["sigma"]))
    if family == "colorcast":   return color_cast(img, r=p["r"], g=p["g"], b=p["b"])
    return img

def prepare_dataset(args):
    in_img = Path(args.input_images)
    in_lbl = Path(args.input_labels)
    out_img = Path(args.output_images)
    out_lbl = Path(args.output_labels)

    if args.clear:
        if out_img.parent.exists(): shutil.rmtree(out_img.parent)
    out_img.mkdir(parents=True, exist_ok=True)
    if not out_lbl.exists():
        shutil.copytree(in_lbl, out_lbl)
    else:
        # sync labels if needed
        pass

    params = json.loads(args.params)
    random.seed(args.seed); np.random.seed(args.seed)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}
    images = sorted([p for p in in_img.rglob("*") if p.suffix.lower() in exts])

    # base knobs
    b_bright = float(params.get("b_bright", 1.0))
    b_blur   = int(params.get("b_blur", 0))
    b_noise  = float(params.get("b_noise", 0.0))

    for p in tqdm(images, desc=f"{args.family} ({'hue' if args.hue else 'no-hue'})"):
        img = Image.open(p).convert("RGB")

        if b_blur > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=b_blur))
        if abs(b_bright - 1.0) > 1e-3:
            img = ImageEnhance.Brightness(img).enhance(b_bright)
        if args.hue:
            img = jitter_hue(img, 30.0)
        if b_noise > 0:
            img = Image.fromarray(add_gaussian_noise(np.array(img), 0.0, b_noise))

        img = apply_family(img, args.family, params)

        out = out_img / p.relative_to(in_img)
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-images", required=True, help="Path to original validation images")
    ap.add_argument("--input-labels", required=True, help="Path to original validation labels")
    ap.add_argument("--output-images", required=True, help="Path to output corrupted images")
    ap.add_argument("--output-labels", required=True, help="Path to output labels (copied from input)")
    ap.add_argument("--family", required=True,
                    choices=["lowpass","downup","motion","fog","jpeg","cutout","vignette","posterize","poisson","colorcast"],
                    help="Corruption family to apply")
    ap.add_argument("--params", required=True, help="JSON dict of parameters for the chosen family (also supports b_bright/b_blur/b_noise).")
    ap.add_argument("--hue", action="store_true", help="Apply random hue jitter")
    ap.add_argument("--seed", type=int, default=123, help="Random seed")
    ap.add_argument("--clear", action="store_true", help="Clear output directory before writing")
    args = ap.parse_args()
    prepare_dataset(args)
