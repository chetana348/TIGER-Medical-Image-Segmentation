import json
import numpy as np
from pycocotools import mask  # kept to match your imports (unused)
import cv2
import os
import sys
import io
import tifffile as tiff

if sys.version_info[0] >= 3:
    unicode = str

caption = "a MRI slice showing the prostate gland which is below the bright uterus."
#"a pkd kidney T1-weighted arterial contrast-enhanced MRI showing a suspicious tumor region."
#"a pkd kidney ktrans slice containing a suspicious tumor region which is usually darker than the neighboring pixels."
phrase = "MRI prostate" # pkd kidney tumor"
tokens_positive = [[2, 4]]

def maskTobbox(ground_truth_binary_mask, image_name):
    contours, _ = cv2.findContours(
        ground_truth_binary_mask* 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    bbox = []
    if len(contours) == 0:
        print("0")
    for contour in contours:
        if len(contour)<3:
            print(image_name)
            print("The contour does not constitute an area")
            continue
        x0, y0, w, h = cv2.boundingRect(contour)
        x1 = x0 + w
        y1 = y0 + h
        bbox.append([x0, y0, x1, y1])
    return bbox


# ------------------ PATHS --------------------
block_mask_path = '/users/PAS3110/sephora20/workspace/PDAC/data/pros/train/labels/'
block_mask_image_files = sorted(os.listdir(block_mask_path))

jsonPath = '/users/PAS3110/sephora20/workspace/PDAC/data/pros/train/labels/train_odvg.json'
path = '/users/PAS3110/sephora20/workspace/PDAC/data/pros/train/images/'

rgb_image_files = sorted(os.listdir(path))
print(len(rgb_image_files), len(block_mask_image_files))
if block_mask_image_files != rgb_image_files:
    print("error")

# ------------------ WRITE ODVG PER IMAGE ----------------------
with io.open(jsonPath, 'w', encoding='utf8') as output:
    for image_name in rgb_image_files:
        if not os.path.exists(os.path.join(block_mask_path, image_name)):
            continue

        # --- image: use tifffile for shape ---
        try:
            img_arr = tiff.imread(os.path.join(path, image_name))
        except Exception as e:
            print("[WARN] failed to read image:", image_name, e)
            continue

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
            m = tiff.imread(os.path.join(block_mask_path, image_name))
        except Exception as e:
            print("[WARN] failed to read mask:", image_name, e)
            continue

        if m.ndim > 2:
            m = m[..., 0] if m.shape[-1] < 8 else m[0]
        m = m.astype(np.float32)
        if m.max() > 1.0:
            denom = m.max() if m.max() > 0 else 1.0
            m = m / denom
        mask_image = (m > 0).astype(np.uint8)

        # get bboxes
        bboxes = maskTobbox(mask_image, image_name)

        # detection.instances
        instances = []
        for i, box in enumerate(bboxes):
            instances.append({
                "bbox": box,
                "label": i,
                "category": "prostate"
            })

        # grounding.regions
        regions = []
        for box in bboxes:
            regions.append({
                "bbox": box,
                "phrase": phrase,
                "tokens_positive": tokens_positive
            })

        annotation = {
            "filename": image_name,
            "height": int(h),
            "width": int(w),
            "detection": {
                "instances": instances
            },
            "grounding": {
                "caption": caption,
                "regions": regions
            }
        }

        # one JSON object per line
        output.write(unicode(json.dumps(annotation)))
        output.write(unicode('\n'))

print(f"Done. Wrote ODVG records to {jsonPath}")
