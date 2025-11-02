import os, json, random
from pathlib import Path

# === CONFIG ===
images_dir = Path("/users/PAS3110/sephora20/workspace/PDAC/data/pkd/val/images")
out_dir    = Path("/users/PAS3110/sephora20/workspace/PDAC/data/pkd/val")
out_dir.mkdir(parents=True, exist_ok=True)
EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# --- Polycystic Kidney Disease (PKD) Prompt Banks ---

organ_terms = [
    "bilateral renal",
    "polycystic kidney",
    "enlarged renal",
    "kidney parenchymal",
    "renal cortical and medullary",
]

scan_terms = [
    "T2-weighted MRI slice",
    "T1-weighted MRI slice",
    "contrast-enhanced MRI image",
    "non-contrast anatomical MRI frame",
    "clinical abdominal MRI cross-sectional slice",
]

pkd_terms = [
    "multiple fluid-filled cysts",
    "numerous variably sized round cystic lesions",
    "clustered renal cyst formations",
    "diffusely distributed renal cysts",
    "moderate polycystic disease pattern",
]

appearance_terms = [
    "appearing hyperintense relative to the surrounding renal parenchyma",
    "with bright fluid signal intensity that distinguishes cystic spaces",
    "causing expansion and distortion of normal renal architecture",
    "with intervening parenchyma visibly thinned between cyst clusters",
    "resulting in lobulated kidney contour and heterogeneous signal distribution",
    "showing irregular spacing of cysts and mild renal enlargement",
]

def make_prompt():
    return (
        f"{random.choice(organ_terms).capitalize()} "
        f"{random.choice(scan_terms)} demonstrating "
        f"{random.choice(pkd_terms)} "
        f"{random.choice(appearance_terms)}."
    )

# --- collect image stems ---
stems = sorted([
    p.stem for p in images_dir.iterdir()
    if p.is_file() and p.suffix.lower() in EXTS
])
if not stems:
    raise RuntimeError(f"No images found in {images_dir}")

random.seed(1234)  # reproducible
prompt_dict = {stem: make_prompt() for stem in stems}

# --- write outputs ---
txt_path  = out_dir / "prompts.txt"
json_path = out_dir / "prompts.json"

with txt_path.open("w", encoding="utf-8") as f:
    for stem, text in prompt_dict.items():
        f.write(f"{stem}: {text}\n")

with json_path.open("w", encoding="utf-8") as f:
    json.dump(prompt_dict, f, ensure_ascii=False, indent=2)

print(f"Generated {len(stems)} PKD MRI prompts")
print(f" - {txt_path}")
print(f" - {json_path}")
