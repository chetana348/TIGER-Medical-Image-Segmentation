import json
import numpy as np
from pycocotools import mask  # kept to match your imports (unused)
import cv2
import os
import sys
import io
import tifffile as tiff
import random

# ------------------ CONFIG --------------------
# keep a fixed phrase for all regions
PHRASE = "pancreas tumor"
# optional: make captions deterministic for a run
random.seed(42)

if sys.version_info[0] >= 3:
    unicode = str

# ------------------ PROMPT BANKS (no location terms) --------------------
organ_terms = [
    "pancreatic", "pancreatic parenchymal", "pancreas-associated",
    "pancreatic soft tissue", "pancreatic glandular", "pancreatic ductal",
    "pancreatic tissue", "pancreatic region", "pancreatic anatomical",
    "pancreatic structural", "pancreatic microvascular",
]

scan_terms = [
    "Ktrans slice", "Ktrans parametric slice", "Ktrans parametric map",
    "Ktrans quantitative map", "Ktrans permeability map", "Ktrans perfusion map",
    "Ktrans-derived parametric image", "Ktrans-based perfusion parameter map",
]

tumor_terms = [
    "suspicious tumor", "abnormal lesion", "hypoenhancing mass",
    "contrast-deficient region", "darkened pathological zone",
]

appearance_terms = [
    "appearing darker than surrounding tissue",
    "darker relative to adjacent parenchyma",
    "with visually diminished signal intensity",
    "showing reduced contrast washout",
    "visibly hypointense compared to background",
]

# ------------------ CAPTION HELPERS --------------------
def choose_lead():
    return random.choice([
        "A", "This", "In this", "On this"
    ])

def choose_opening():
    # e.g., "A pancreatic Ktrans parametric map depicts"
    return f"{choose_lead()} {random.choice(organ_terms)} {random.choice(scan_terms)} {random.choice(['shows','depicts','contains','demonstrates'])}"

def area_bucket(mask_image):
    a = float(mask_image.sum())
    if a <= 0:
        return "no evident extent"
    pct = 100.0 * a / mask_image.size
    if pct < 0.2:  return "very small in extent"
    if pct < 1.0:  return "small in extent"
    if pct < 3.0:  return "moderate in extent"
    if pct < 8.0:  return "large in extent"
    return "very large in extent"

def local_intensity_contrast(img, mask_image):
    # compare mean inside vs a thin ring around the mask
    if img.ndim == 3:
        # assume last dim is channel, convert to gray for metric
        if img.shape[-1] in (3, 4):
            gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
        else:
            gray = img[..., 0].astype(np.float32)
    else:
        gray = img.astype(np.float32)

    m = (mask_image > 0).astype(np.uint8)
    if m.sum() == 0:
        return "with indeterminate intensity relative to the background"

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    dil = cv2.dilate(m, kernel, iterations=1)
    ring = (dil - m).astype(bool)

    mean_in  = float(gray[m.astype(bool)].mean()) if m.any() else np.nan
    mean_out = float(gray[ring].mean()) if ring.any() else np.nan
    if not np.isfinite(mean_in) or not np.isfinite(mean_out):
        return "with indeterminate intensity relative to the background"

    diff = mean_in - mean_out
    s = float(gray.std()) + 1e-6
    if diff < -0.25 * s:
        return random.choice(appearance_terms)  # darker set already
    if diff >  0.25 * s:
        return "appearing brighter than surrounding tissue"
    return "with similar intensity to adjacent parenchyma"

def texture_note(img, mask_image):
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):
            gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
        else:
            gray = img[..., 0].astype(np.float32)
    else:
        gray = img.astype(np.float32)

    m = (mask_image > 0).astype(bool)
    if not m.any():
        return ""
    vals = gray[m]
    if vals.size < 10:
        return ""
    rel = float(vals.std()) / (float(gray.std()) + 1e-6)
    if rel > 1.2:
        return " with heterogeneous internal texture"
    if rel < 0.7:
        return " with relatively homogeneous texture"
    return ""

def caption_and_tokens(img, mask_image, phrase=PHRASE):
    """
    Build a per-image caption WITHOUT location words.
    Ensures the exact phrase 'pancreas tumor' appears once and returns token indices.
    """
    opening = choose_opening()

    if mask_image.sum() == 0:
        # no tumor → do not insert the phrase (tokens_positive = [])
        caption = f"{opening} no clear {phrase}."
        return caption, []

    size_clause = area_bucket(mask_image)
    contrast_clause = local_intensity_contrast(img, mask_image)
    texture_clause = texture_note(img, mask_image)

    # pick a tumor synonym lead-in but still include exact phrase once
    tumor_lead = random.choice(tumor_terms)

    # Construct caption. Example:
    # "A pancreatic Ktrans parametric map shows a suspicious tumor (pancreas tumor) region,
    # appearing darker..., small in extent with heterogeneous internal texture."
    # We insert the exact phrase once to compute tokens reliably.
    caption = (
        f"{opening} a {tumor_lead} ({phrase}) region, "
        f"{contrast_clause}, {size_clause}{texture_clause}."
    )

    # Compute tokens_positive for the exact two-token phrase "pancreas tumor"
    words = caption.replace(",", "").replace(".", "").split()
    start_idx = None
    for i in range(len(words) - 1):
        if words[i].lower() == "pancreas" and words[i+1].lower() == "tumor":
            start_idx = i
            break
    tokens_positive = [] if start_idx is None else [[start_idx, start_idx + 1]]
    return caption, tokens_positive

# ------------------ GEOMETRY --------------------
def maskTobbox(ground_truth_binary_mask):
    contours, _ = cv2.findContours(
        ground_truth_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    bbox = []
    if len(contours) == 0:
        print("0")
    for contour in contours:
        if len(contour) < 3:
            print("The contour does not constitute an area")
            continue
        x0, y0, w, h = cv2.boundingRect(contour)
        x1 = x0 + w
        y1 = y0 + h
        bbox.append([x0, y0, x1, y1])
    return bbox

# ------------------ PATHS --------------------
block_mask_path = '/users/PAS3110/sephora20/workspace/PDAC/data/train/labels/'
jsonPath = '/users/PAS3110/sephora20/workspace/PDAC/data/train/labels/train_odvg_mul.json'
path = '/users/PAS3110/sephora20/workspace/PDAC/data/train/images/'

block_mask_image_files = sorted([
    f for f in os.listdir(block_mask_path)
    if f.lower().endswith('.tif')
])

rgb_image_files = sorted([
    f for f in os.listdir(path)
    if f.lower().endswith('.tif')
])
print(len(rgb_image_files), len(block_mask_image_files))
if block_mask_image_files != rgb_image_files:
    print("error")

# ------------------ WRITE ODVG PER IMAGE ----------------------
with io.open(jsonPath, 'w', encoding='utf8') as output:
    for image_name in rgb_image_files:
        mask_fp = os.path.join(block_mask_path, image_name)
        img_fp  = os.path.join(path, image_name)
        if not os.path.exists(mask_fp):
            continue

        # --- image: use tifffile for shape ---
        try:
            img_arr = tiff.imread(img_fp)
        except Exception as e:
            print("[WARN] failed to read image:", image_name, e)
            continue

        # normalize shape handling
        if img_arr.ndim == 3 and img_arr.shape[-1] not in (1, 3, 4):
            img_arr = img_arr[0]
        if img_arr.ndim == 2:
            h, w = img_arr.shape
        elif img_arr.ndim == 3:
            h, w = img_arr.shape[:2]
        else:
            h, w = img_arr.shape[-2], img_arr.shape[-1]

        # --- mask: use tifffile, binarize ---
        try:
            m = tiff.imread(mask_fp)
        except Exception as e:
            print("[WARN] failed to read mask:", image_name, e)
            continue

        if m.ndim > 2:
            m = m[..., 0] if m.shape[-1] < 8 else m[0]
        m = m.astype(np.float32)
        if m.max() > 1.0:
            denom = m.max() if m.max() > 0 else 1.0
            m = m / denom
        mask_image = (m >= 0.5).astype(np.uint8)

        # --- get bboxes ---
        bboxes = maskTobbox(mask_image)

        # --- detection.instances ---
        instances = []
        for i, box in enumerate(bboxes):
            instances.append({
                "bbox": box,
                "label": i,
                "category": "pancreas"
            })

        # --- caption & tokens (varies per image) ---
        caption, tokens_positive = caption_and_tokens(img_arr, mask_image, phrase=PHRASE)

        # --- grounding.regions (same phrase for all regions) ---
        regions = []
        for box in bboxes:
            regions.append({
                "bbox": box,
                "phrase": PHRASE,
                "tokens_positive": tokens_positive
            })

        annotation = {
            "filename": image_name,
            "height": int(h),
            "width": int(w),
            "detection": {"instances": instances},
            "grounding": {
                "caption": caption,      # <-- per-image unique caption
                "regions": regions       # <-- all regions share the same phrase
            }
        }

        # one JSON object per line
        output.write(unicode(json.dumps(annotation)))
        output.write(unicode('\n'))

print(f"Done. Wrote ODVG records to {jsonPath}")
