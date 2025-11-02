import os
import numpy as np
import tifffile as tiff
from PIL import Image
from glob import glob

images_dir = "/users/PAS3110/sephora20/workspace/PDAC/data/pros/train/images"
latents_dir = "/users/PAS3110/sephora20/workspace/PDAC/data/pros/train/pseudo_latents/latents_soft"
out_dir = "/users/PAS3110/sephora20/workspace/PDAC/data/pros/train/images_soft"
os.makedirs(out_dir, exist_ok=True)

# ---------- 1) Robust dataset-level scaling for latents ----------
latent_files = sorted(glob(os.path.join(latents_dir, "*_soft.tif")))
if len(latent_files) == 0:
    raise FileNotFoundError(f"No *_soft.tif files in {latents_dir}")

# sample (all or a subset) to compute 1st/99th percentiles
samples = []
for i, lp in enumerate(latent_files):
    if i % max(1, len(latent_files)//200) != 0 and len(latent_files) > 200:
        continue  # subsample ~<=200 files for speed
    with Image.open(lp) as L:
        arr = np.array(L)
    if arr.ndim == 3:  # squeeze if (H,W,1)
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    samples.append(arr.ravel())
samples = np.concatenate(samples)
p1, p99 = np.percentile(samples, [1, 99])
if p99 <= p1:  # safety
    p1, p99 = float(samples.min()), float(samples.max())
    if p99 == p1:
        p1, p99 = 0.0, 1.0
print(f"[latent scaling] p1={p1:.6g}, p99={p99:.6g}")

def norm_image_uint8(x):
    x = x.astype(np.float32)
    if x.max() > 1.0:  # typical uint8 0..255
        x = x / 255.0
    else:
        # already [0,1] or float – leave as is
        x = np.clip(x, 0.0, 1.0)
    return x

def norm_latent(x):
    x = x.astype(np.float32)
    # dataset-level robust scaling to [0,1]
    x = (x - p1) / (p99 - p1 + 1e-8)
    x = np.clip(x, 0.0, 1.0)
    return x

# ---------- 2) Convert & save ----------
for img_name in os.listdir(images_dir):
    if not img_name.lower().endswith(".tif"):
        continue
    base = os.path.splitext(img_name)[0]
    latent_name = f"{base}_soft.tif"

    img_path = os.path.join(images_dir, img_name)
    latent_path = os.path.join(latents_dir, latent_name)

    if not os.path.exists(latent_path):
        print(f"⚠️ No matching latent for {img_name}, skipping.")
        continue

    # Load
    with Image.open(img_path) as I:
        img = np.array(I)  # (H,W) uint8/uint16
    with Image.open(latent_path) as L:
        lat = np.array(L)  # (H,W) float/uint

    # Resize latent to match image size (PIL resizes need PIL.Image)
    if (lat.shape[1], lat.shape[0]) != (img.shape[1], img.shape[0]):
        L = Image.fromarray(lat)
        L = L.resize((img.shape[1], img.shape[0]), resample=Image.BILINEAR)
        lat = np.array(L)

    # If they have extra dims, squeeze to (H,W)
    if img.ndim == 3:
        # take single channel if grayscale stored as (H,W,1) or average RGB if it slipped in
        if img.shape[2] == 1:
            img = img[..., 0]
        else:
            img = img.mean(axis=2)
    if lat.ndim == 3 and lat.shape[2] == 1:
        lat = lat[..., 0]

    # Normalize per channel
    img_n = norm_image_uint8(img)    # -> float32 [0,1]
    lat_n = norm_latent(lat)         # -> float32 [0,1]

    # Stack (C,H,W) float32
    combined = np.stack([img_n, lat_n], axis=0).astype(np.float32)

    # Save as compressed BigTIFF with channel axis metadata
    out_path = os.path.join(out_dir, base + ".tif")
    tiff.imwrite(
        out_path,
        combined,
        dtype=np.float32,
        bigtiff=True,
        compression="zlib",
        photometric="minisblack",
        metadata={"axes": "CYX"}  # channel, y, x
    )
    print(f"Saved: {out_path} | shape={combined.shape}, dtype={combined.dtype}")
