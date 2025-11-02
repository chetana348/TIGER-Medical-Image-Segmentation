#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build COCO annotations (val.json) from binary masks via contours.

Expected layout:
  /users/PAS3110/sephora20/workspace/PDAC/data/val/
    images/
      *.tif
    labels/            # change LABELS_DIR if yours is "masks"
      *.tif

Output:
  /users/PAS3110/sephora20/workspace/PDAC/data/val/val.json
"""

import json
from pathlib import Path
import numpy as np
import cv2
import tifffile as tiff
from datetime import datetime

# ------------------ CONFIG ------------------
ROOT = Path("/users/PAS3110/sephora20/workspace/PDAC/data/pros/val")
IMAGES_DIR = ROOT / "images"
LABELS_DIR = ROOT / "labels"     # set to ROOT/"masks" if that's your folder
OUT_JSON   = ROOT / "labels" / "val.json"

CATEGORY_NAME = "prostate" #"pancreas"       # single-class; id=1
THRESH_BIN = 100                 # threshold for masks if they are 0..255; ignored if we normalize
MIN_CONTOUR_AREA = 1             # drop tiny speckles (pixels)
ALLOWED_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# If your labels are not strictly 0/255, set NORMALIZE_MASK=True to normalize to [0,1] then >=0.5
NORMALIZE_MASK = True

# --------------------------------------------

def list_files(d: Path):
    return sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS])

def read_image_hw(path: Path):
    """Return (H,W) for a TIFF/PNG/JPG using tifffile for robustness."""
    arr = tiff.imread(str(path))
    # handle multi-page stacks (Z,H,W[,C]) -> first page
    if arr.ndim == 3 and arr.shape[-1] not in (1, 3, 4):
        arr = arr[0]
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        h, w = arr.shape[:2]
    else:
        h, w = arr.shape[-2], arr.shape[-1]
    return int(h), int(w)

def read_mask_binary(path: Path):
    """Read mask and binarize to {0,1} uint8."""
    m = tiff.imread(str(path))
    if m.ndim > 2:
        # prefer last channel if reasonable, otherwise first slice
        m = m[..., 0] if m.shape[-1] < 8 else m[0]
    m = m.astype(np.float32)
    if NORMALIZE_MASK:
        mx = m.max()
        if mx > 0:
            m = m / mx
        m = (m >= 0.5).astype(np.uint8)
    else:
        # assume 0..255
        _, m = cv2.threshold(m.astype(np.uint8), THRESH_BIN, 1, cv2.THRESH_BINARY)
    return m

def mask_to_polygons_and_boxes(bin_mask: np.ndarray):
    """Return list of (segmentation_flat, area, bbox_xywh)."""
    m255 = (bin_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(m255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        area = float(cv2.contourArea(cnt))
        if area < MIN_CONTOUR_AREA:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        # COCO expects segmentation as a list of lists of x,y pairs flattened
        # Ensure clockwise ordering from cv2 is acceptable (it is).
        cnt = np.squeeze(cnt, axis=1)  # Nx2
        seg = cnt.reshape(-1).astype(float).tolist()
        if len(seg) >= 6:  # at least 3 points
            results.append((seg, area, [int(x), int(y), int(w), int(h)]))
    return results

def build_coco(images_dir: Path, labels_dir: Path, out_json: Path):
    # filenames intersection
    imgs = {p.name: p for p in list_files(images_dir)}
    msks = {p.name: p for p in list_files(labels_dir)}
    common = sorted(set(imgs.keys()) & set(msks.keys()))
    if not common:
        raise RuntimeError("No matching image/label filenames found.")

    # Info / licenses / categories
    coco = {
        "info": {
            "year": datetime.now().year,
            "version": "1.0",
            "description": "PDAC val annotations",
            "contributor": "PDAC-SimtxtSeg",
            "date_created": datetime.now().strftime("%Y-%m-%d")
        },
        "licenses": [{
            "id": 1,
            "name": "CC BY 4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/"
        }],
        "categories": [{
            "id": 1,
            "name": CATEGORY_NAME,
            "supercategory": CATEGORY_NAME
        }],
        "images": [],
        "annotations": []
    }

    # Build images with ids, and a map filename->image_id
    filename_to_id = {}
    for idx, name in enumerate(common, start=1):
        h, w = read_image_hw(imgs[name])
        img_rec = {
            "id": idx,
            "file_name": name,
            "height": h,
            "width": w
        }
        coco["images"].append(img_rec)
        filename_to_id[name] = idx

    # Build annotations
    ann_id = 1
    for name in common:
        img_id = filename_to_id[name]
        bin_mask = read_mask_binary(msks[name])
        poly_boxes = mask_to_polygons_and_boxes(bin_mask)
        for seg, area, bbox in poly_boxes:
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "iscrowd": 0,
                "area": area,
                "bbox": bbox,              # [x,y,w,h]
                "segmentation": [seg]      # list of one polygon
            }
            coco["annotations"].append(ann)
            ann_id += 1

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_json} | images: {len(coco['images'])}, anns: {len(coco['annotations'])}")

if __name__ == "__main__":
    build_coco(IMAGES_DIR, LABELS_DIR, OUT_JSON)
