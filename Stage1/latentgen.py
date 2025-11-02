# -*- coding: utf-8 -*-
"""
Detect -> (Boxes) -> SAM -> (Masks) + Latents
Adds export of second-channel latents for training:
  - soft:     SAM low-res logits (sigmoid) upsampled to (H,W)
  - embednorm:||E|| across embedding channels upsampled to (H,W)
  - embedK:   first K embedding channels upsampled to (H,W,K)
  - detheat:  detector box-score heatmap (H,W)

Outputs:
  - out_dir/<base>.png                         (merged/added instance mask as before)
  - out_dir/<base>_mask_#.png                 (per-instance masks, cleaned after merge)
  - out_dir/overlay/<base>.png                (optional overlay)
  - out_dir/latents_*/*.tif                   (float32 TIFFs for each latent type)
"""

import argparse
import os
import ast
import math
import sys
from argparse import Namespace
from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import tifffile as tiff  # for clean float32 multi-channel TIFFs

# mmdet
try:
    import mmdet  # noqa: F401
    from mmdet.apis import inference_detector, init_detector
except ImportError:
    mmdet = None

sys.path.append('../')

from mmengine.config import Config
from mmengine.utils import ProgressBar, scandir

# grounding dino DetInferencer (optional path)
from mmdet.apis.det_inferencer import DetInferencer

import urllib

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
                  '.tiff', '.webp')


def get_file_list(source_root: str):
    """Get list of image files from dir/url/single file."""
    is_dir = os.path.isdir(source_root)
    is_url = source_root.startswith(('http:/', 'https:/'))
    is_file = os.path.splitext(source_root)[-1].lower() in IMG_EXTENSIONS

    source_file_path_list = []
    if is_dir:
        for file in scandir(source_root, IMG_EXTENSIONS, recursive=True):
            source_file_path_list.append(os.path.join(source_root, file))
    elif is_url:
        filename = os.path.basename(
            urllib.parse.unquote(source_root).split('?')[0])
        file_save_path = os.path.join(os.getcwd(), filename)
        print(f'Downloading source file to {file_save_path}')
        torch.hub.download_url_to_file(source_root, file_save_path)
        source_file_path_list = [file_save_path]
    elif is_file:
        source_file_path_list = [source_root]
    else:
        print('Cannot find image file.')

    source_type = dict(is_dir=is_dir, is_url=is_url, is_file=is_file)
    return source_file_path_list, source_type


def parse_args():
    parser = argparse.ArgumentParser('Detect-Segment-Anything (GroundingDINO + SAM) with Latents')
    # IO
    parser.add_argument('--image', type=str,
                        default='/users/PAS3110/sephora20/workspace/PDAC/data/pros/train/images/',
                        help='path to image file OR a directory')
    parser.add_argument('--out-dir', '-o', type=str,
                        default='/users/PAS3110/sephora20/workspace/PDAC/data/pros/train/pseudo_latents/',
                        help='output directory')
    parser.add_argument('--batch-size', type=int, default=1)

    # GroundingDINO/MMDet
    parser.add_argument('--det_config', type=str,
                        default='/users/PAS3110/sephora20/workspace/PDAC/Tiger/Stage1/Grounding_config.py',
                        help='path to det config file')
    parser.add_argument('--det_weight', type=str,
                        default='/users/PAS3110/sephora20/workspace/PDAC/Tiger/Stage1/outputs/pros/best_coco_bbox_mAP_epoch_12.pth',
                        help='path to det weight file')
    parser.add_argument('--det-device', '-d', default='cuda:0')
    parser.add_argument('--box-thr', '-b', type=float, default=0.02)
    parser.add_argument('--text-prompt', '-t', type=str,
                        default="a MRI slice showing the prostate gland which is below the bright uterus.",
                        help='text prompt (will be lowercased and period-terminated)')
    parser.add_argument('--text-thr', type=float, default=0.25)
    parser.add_argument('--apply-original-groudingdino', action='store_true', default=True,
                        help='use MMDet GroundingDINO path (recommended)')

    # DetInferencer-only extras
    parser.add_argument('--custom-entities', '-c', action='store_true', default=False,
                        help='custom entity names with "cls1 . cls2 ." format')
    parser.add_argument('--chunked-size', type=int, default=-1,
                        help='truncate predictions if many categories')
    parser.add_argument('--tokens-positive', '-p', type=str, default="[[[38,49]]]",
                        help='interest positions in text; "-1" for none; "None" to ignore')

    # SAM
    parser.add_argument('--sam-type', type=str, default='vit_b', choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument('--sam-weight', type=str,
                        default='/users/PAS3110/sephora20/workspace/PDAC/Tiger/Stage1/pretrained_weights/sam_vit_b_01ec64.pth')
    parser.add_argument('--sam-device', '-s', default='cuda:0')
    parser.add_argument('--sam-multimask', action='store_true', default=False,
                        help='If True, SAM returns multiple masks per box')

    # Visualization
    parser.add_argument('--save-overlay', action='store_true', default=True)

    # Latent exports
    parser.add_argument('--latent-types', type=str,
                        default='soft,embednorm,detheat',
                        help='comma-separated subset of {soft,embednorm,embedK,detheat,none}')
    parser.add_argument('--embed-k', type=int, default=4,
                        help='number of embedding channels to export if embedK is enabled')
    return parser.parse_args()


def build_init_args(args):
    return dict(
        show_progress=False,
        model=args.det_config,
        weights=args.det_weight,
        device=args.det_device,
        palette=None
    )


def build_call_args(args, image_path):
    call_args = dict(
        inputs=image_path,
        texts=args.text_prompt if args.text_prompt.endswith('.') else (args.text_prompt + '.'),
        custom_entities=args.custom_entities,
        pred_score_thr=args.box_thr
    )
    if args.tokens_positive is not None and args.tokens_positive != "None":
        if args.tokens_positive.strip() == "-1":
            call_args['tokens_positive'] = -1
        else:
            call_args['tokens_positive'] = ast.literal_eval(args.tokens_positive)
    return call_args


def build_detector(args):
    config = Config.fromfile(args.det_config)
    detector = init_detector(config, args.det_weight, device=args.det_device, cfg_options={})
    return detector


def build_sam(args):
    """Build SAM or SAM-Med2D predictor and move to device."""
    if "med2d" not in args.sam_weight.lower():
        from segment_anything import SamPredictor, sam_model_registry
        predictor = SamPredictor(sam_model_registry[args.sam_type](checkpoint=args.sam_weight))
    else:
        from segment_anything_med2d import SamPredictor, sam_model_registry
        sam_args = Namespace()
        sam_args.image_size = 256
        sam_args.encoder_adapter = True
        sam_args.sam_checkpoint = args.sam_weight
        predictor = SamPredictor(sam_model_registry[args.sam_type](sam_args))
    predictor.model.to(args.sam_device)
    return predictor


@torch.no_grad()
def run_detector(det_model, image_path, args):
    """
    Returns:
      pred_dict: { 'boxes': Tensor[N,4], 'scores': list[float], 'labels': list[int] }
      meta: optional metainfo
    """
    pred_dict, meta = {}, None
    text_prompt = args.text_prompt.strip().lower()
    if not text_prompt.endswith('.'):
        text_prompt += '.'

    if args.apply_original_groudingdino:
        result = inference_detector(det_model, image_path, text_prompt=text_prompt)
        inst = result.pred_instances
        keep = inst.scores > args.box_thr
        inst = inst[keep]
        pred_dict['boxes'] = inst.bboxes  # (N,4) tensor
        pred_dict['scores'] = inst.scores.detach().cpu().numpy().tolist()
        pred_dict['labels'] = inst.labels.detach().cpu().numpy().tolist()
        meta = getattr(result, 'metainfo', None)
    else:
        inferencer = DetInferencer(**build_init_args(args))
        if args.chunked_size and args.chunked_size > 0:
            inferencer.model.test_cfg.chunked_size = args.chunked_size
        outputs = inferencer(**build_call_args(args, image_path))
        res = outputs['predictions'][0]
        scores = np.array(res['scores'])
        keep = scores > args.box_thr
        boxes = torch.tensor(res['bboxes'])[keep]
        labels = np.array(res['labels'])[keep].tolist()
        pred_dict['boxes'] = boxes
        pred_dict['scores'] = scores[keep].tolist()
        pred_dict['labels'] = labels
        meta = outputs.get('metainfo', None)

    return pred_dict, meta


def draw_and_save(image, pred_dict, save_path, random_color=True, show_label=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    labels = pred_dict.get('labels', [])
    scores = pred_dict.get('scores', [])
    bboxes = pred_dict.get('boxes', torch.empty((0, 4)))
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()

    ax = plt.gca()
    for box, label, score in zip(bboxes, labels, scores):
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
        if show_label:
            ax.text(x0, y0, f'{label}|{round(score, 2)}', color='white', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))

    if 'masks' in pred_dict:
        masks = pred_dict['masks'].detach().cpu().numpy()
        masks = masks.squeeze(1) if masks.ndim == 4 and masks.shape[1] == 1 else masks
        for mask in masks:
            color = (np.random.random(3).tolist() + [0.6]) if random_color else [30/255, 144/255, 255/255, 0.6]
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
            ax.imshow(mask_image)

    plt.axis('off')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def save_instance_masks(masks, path_prefix):
    """Save each instance mask as separate PNG."""
    masks_np = masks.detach().cpu().numpy()
    masks_np = masks_np.squeeze(1) if masks_np.ndim == 4 and masks_np.shape[1] == 1 else masks_np
    for i, mask_np in enumerate(masks_np):
        mask_path = path_prefix + f"_mask_{i+1}.png"
        mask_u8 = (mask_np.astype(np.float32) > 0.5).astype(np.uint8) * 255
        Image.fromarray(mask_u8).save(mask_path)


def make_pair(dataset_dir):
    """Group files that share the same stem, to merge instance masks."""
    pair_dict = defaultdict(list)
    for name in os.listdir(dataset_dir):
        stem = os.path.splitext(name)[0]
        pair_dict[stem].append(os.path.join(dataset_dir, name))
    return pair_dict


def mask_add(image_mask_pair):
    """Merge *_mask_#.png into a single grayscale PNG per image."""
    for values in image_mask_pair.values():
        instance_files = [p for p in values if '_mask_' in os.path.basename(p)]
        base_pngs = [p for p in values if '_mask_' not in os.path.basename(p) and p.lower().endswith('.png')]
        if len(instance_files) == 0:
            continue
        base = instance_files[0].split('_mask_')[0] + ".png"
        result = None
        for p in sorted(instance_files):
            mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not read instance mask {p}")
                continue
            if result is None:
                result = np.zeros_like(mask, dtype=np.uint16)
            result = np.clip(result + mask.astype(np.uint16), 0, 255)
        if result is not None:
            cv2.imwrite(base, result.astype(np.uint8))
        for p in instance_files:
            try:
                os.remove(p)
            except Exception:
                pass
        for p in base_pngs:
            if os.path.abspath(p) != os.path.abspath(base):
                try:
                    os.remove(p)
                except Exception:
                    pass


def norm01(x: np.ndarray):
    m, M = float(x.min()), float(x.max())
    if M > m:
        return (x - m) / (M - m)
    return np.zeros_like(x, dtype=np.float32)


def _to_nchw_lowres(low_res) -> torch.Tensor:
    """
    Normalize 'low_res' into a float tensor shaped (N,1,h,w) on CPU.
    Accepts: torch.Tensor, list/tuple of tensors, or nested lists.
    Handles shapes: (N,1,H,W), (N,H,W), (N,L), (H,W), (L), (1,L), (1,H,W), etc.
    Tries to infer h,w by squaring when needed (e.g., L==256 -> 16x16).
    Returns None if impossible.
    """
    if low_res is None:
        return None

    # Unwrap containers
    if isinstance(low_res, (list, tuple)):
        # flatten one level and filter non-tensors
        items = []
        for x in low_res:
            if isinstance(x, (list, tuple)):
                items += [y for y in x if isinstance(y, torch.Tensor)]
            elif isinstance(x, torch.Tensor):
                items.append(x)
        if len(items) == 0:
            return None
        # Stack if possible: first make each item (1,1,h,w)
        normed = []
        for t in items:
            t = _to_nchw_lowres(t)
            if t is None:
                continue
            # _to_nchw_lowres returns (1,1,h,w) for single tensors
            if t.ndim == 4 and t.shape[0] == 1:
                normed.append(t)
            else:
                # If it accidentally returned (N,1,h,w), split batch
                normed += [ti.unsqueeze(0) for ti in t]
        if len(normed) == 0:
            return None
        return torch.cat(normed, dim=0).float().cpu()

    # Tensor path
    if not isinstance(low_res, torch.Tensor):
        return None
    t = low_res.detach()
    # Move to CPU to avoid device mismatches downstream
    t = t.float().cpu()

    # Remove singleton batch/channel dims where obvious
    # Common shapes we may see here:
    # (N,1,H,W), (N,H,W), (H,W), (N,L), (L), (1,L), (1,H,W), (N,1,H), (N,H,1)
    if t.ndim == 4:
        N, C, H, W = t.shape
        if C == 1:
            return t  # already (N,1,H,W)
        # Reduce multi-channel logits by max across channels
        return torch.amax(t, dim=1, keepdim=True)
    elif t.ndim == 3:
        a, b, c = t.shape
        # Heuristics:
        # (N,H,W) -> add channel
        # (1,H,W) -> treat as single N
        if a == 1 and b > 1 and c > 1:
            return t.unsqueeze(1)  # (1,1,H,W)
        if a > 1 and b > 1 and c > 1:
            return t.unsqueeze(1)  # (N,1,H,W)
        # (N,1,L) or (N,L,1) -> try to square L
        if a > 0 and b == 1 and c > 1:
            N, _, L = t.shape
            s = int(math.isqrt(int(L)))
            if s * s == L:
                return t.view(N, 1, s, s)
        if a > 0 and b > 1 and c == 1:
            N, L, _ = t.shape
            s = int(math.isqrt(int(L)))
            if s * s == L:
                return t.view(N, 1, s, s)
        return None
    elif t.ndim == 2:
        # (N,L) -> try square
        N, L = t.shape
        s = int(math.isqrt(int(L)))
        if s * s == L:
            return t.view(N, 1, s, s)
        return None
    elif t.ndim == 1:
        # (L,) -> single instance, try square
        L = t.shape[0]
        s = int(math.isqrt(int(L)))
        if s * s == L:
            return t.view(1, 1, s, s)
        return None
    else:
        return None


def export_soft_latent(low_res_logits, out_h: int, out_w: int, save_path: str) -> bool:
    """
    Convert SAM low-res logits to (H,W) float32 TIFF robustly.
    Returns True on success; False to let caller use hard-mask fallback.
    """
    t = _to_nchw_lowres(low_res_logits)  # (N,1,h,w) or None
    # Debug once
    if not hasattr(export_soft_latent, "_dbg"):
        try:
            print("DEBUG low_res raw type/ndim:", type(low_res_logits),
                  getattr(low_res_logits, "ndim", None),
                  "normed shape:" if t is not None else "normed: None",
                  tuple(t.shape) if t is not None else None)
        except Exception:
            pass
        export_soft_latent._dbg = True

    if t is None or t.ndim != 4 or t.shape[1] != 1:
        return False

    # logits -> prob, instance-merge
    soft = torch.sigmoid(t)  # (N,1,h,w)
    
    # Merge instances but KEEP 4D for interpolate
    # (reduce N to 1, but keepdim=True so shape stays (1,1,h,w))
    soft_merged = torch.amax(soft, dim=0, keepdim=True)  # (1,1,h,w)
    
    # Safety: if something upstream collapses dims, re-expand to 4D
    if soft_merged.ndim == 2:                     # (h,w) -> (1,1,h,w)
        soft_merged = soft_merged.unsqueeze(0).unsqueeze(0)
    elif soft_merged.ndim == 3:                   # (1,h,w) or (C,h,w) -> (1,1,h,w)
        soft_merged = soft_merged.unsqueeze(0) if soft_merged.shape[0] != 1 else soft_merged.unsqueeze(1)
    
    # Now upsample with a proper 4D tensor
    soft_up = F.interpolate(
        soft_merged, size=(out_h, out_w),
        mode='bilinear', align_corners=False
    )  # (1,1,H,W)
    
    soft_up = soft_up.squeeze(0).squeeze(0)  # -> (H,W)
    tiff.imwrite(save_path, soft_up.numpy().astype(np.float32))
    return True


def get_sam_embedding(predictor):
    """Try public API first, then internal cache."""
    img_embed = None
    if hasattr(predictor, 'get_image_embedding'):
        try:
            img_embed = predictor.get_image_embedding()
        except Exception:
            img_embed = None
    if img_embed is None:
        # Official repo commonly stores under _features["image_embed"]
        if hasattr(predictor, '_features') and isinstance(predictor._features, dict):
            img_embed = predictor._features.get('image_embed', None)
    return img_embed  # (1,C,H',W') or None


def export_embednorm_and_embedK(img_embed: torch.Tensor, out_h: int, out_w: int,
                                save_dir: str, base_stem: str, K: int):
    """
    img_embed: (1,C,H',W')
    Writes:
      - latents_embed/embednorm/<base>_embednorm.tif  (H,W)
      - latents_embed/embedK/<base>_embed{K}c.tif     (H,W,K) if K>0
    """
    if img_embed is None:
        return

    # ||E||_2 across channels
    embed_norm = torch.linalg.vector_norm(img_embed, dim=1, keepdim=True)  # (1,1,H',W')
    embed_norm_up = F.interpolate(embed_norm, size=(out_h, out_w),
                                  mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    dir_en = os.path.join(save_dir, "embednorm")
    os.makedirs(dir_en, exist_ok=True)
    tiff.imwrite(os.path.join(dir_en, base_stem + "_embednorm.tif"),
                 embed_norm_up.detach().cpu().numpy().astype(np.float32))

    # First K channels
    if K and K > 0:
        k = int(min(K, img_embed.shape[1]))
        if k > 0:
            embed_k = img_embed[:, :k]  # (1,k,H',W')
            embed_k_up = F.interpolate(embed_k, size=(out_h, out_w),
                                       mode='bilinear', align_corners=False).squeeze(0)  # (k,H,W)
            # Save as (H,W,k)
            arr = embed_k_up.permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            dir_ek = os.path.join(save_dir, "embedK")
            os.makedirs(dir_ek, exist_ok=True)
            tiff.imwrite(os.path.join(dir_ek, base_stem + f"_embed{k}c.tif"), arr)


def export_det_heatmap(boxes, scores, out_h: int, out_w: int, save_path: str):
    """
    boxes: (N,4) in original image coords; torch.Tensor | np.ndarray | list
    scores: length-N iterable (or None)
    Writes float32 (H,W) normalized heatmap.
    """
    # --- Normalize boxes to np.ndarray (N,4) ---
    if boxes is None:
        b = np.zeros((0, 4), dtype=np.float32)
    elif isinstance(boxes, torch.Tensor):
        b = boxes.detach().cpu().numpy()
    else:
        b = np.array(boxes)

    if b.size == 0:
        b = b.reshape(0, 4)
    elif b.ndim == 1:
        # e.g., shape (4,)
        b = b.reshape(-1, 4)
    elif b.ndim > 2:
        b = b.reshape(-1, 4)

    # --- Normalize scores to 1D np.ndarray ---
    if scores is None:
        s = np.ones((b.shape[0],), dtype=np.float32)
    else:
        s = np.asarray(scores, dtype=np.float32).ravel()

    # Align lengths (truncate to min)
    n = min(len(b), len(s))
    if n == 0:
        det_heat = np.zeros((out_h, out_w), dtype=np.float32)
        tiff.imwrite(save_path, det_heat)
        return

    b = b[:n]
    s = s[:n]

    # --- Rasterize weighted boxes ---
    det_heat = np.zeros((out_h, out_w), dtype=np.float32)
    for (x0, y0, x1, y1), sc in zip(b, s):
        # clip & int
        x0 = int(max(0, np.floor(x0)))
        y0 = int(max(0, np.floor(y0)))
        x1 = int(min(out_w, np.ceil(x1)))
        y1 = int(min(out_h, np.ceil(y1)))
        if x1 > x0 and y1 > y0:
            det_heat[y0:y1, x0:x1] += float(sc)

    # Smooth & normalize
    if det_heat.max() > 0:
        det_heat = cv2.GaussianBlur(det_heat, (0, 0), sigmaX=3, sigmaY=3)
        m = det_heat.max()
        if m > 0:
            det_heat = det_heat / m

    tiff.imwrite(save_path, det_heat.astype(np.float32))



def main():
    args = parse_args()

    # Parse latent types
    latent_flags = set([t.strip().lower() for t in args.latent_types.split(',') if t.strip() != ''])
    if 'none' in latent_flags:
        latent_flags = set()

    # Build detector
    if args.apply_original_groudingdino:
        det_model = build_detector(args)
    else:
        det_model = None  # created inside run_detector when needed

    # Build SAM predictor
    sam_predictor = build_sam(args)

    # IO
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    files, _ = get_file_list(args.image)
    progress_bar = ProgressBar(len(files))

    # Latent subdirs (created lazily)
    latent_soft_dir = os.path.join(out_dir, "latents_soft")
    latent_embed_root = os.path.join(out_dir, "latents_embed")
    latent_det_dir = os.path.join(out_dir, "latents_det")

    for image_path in files:
        # Load original RGB (for SAM + overlay)
        orig_image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if orig_image_bgr is None:
            print(f"Warning: cannot read {image_path}")
            progress_bar.update()
            continue
        orig_image = cv2.cvtColor(orig_image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = orig_image.shape[:2]

        # Run detector on ORIGINAL image path
        pred_dict, meta = run_detector(det_model, image_path, args)

        # Prepare output paths
        base_name = os.path.basename(image_path)
        base_stem = os.path.splitext(base_name)[0]
        save_path = os.path.join(out_dir, base_name)  # final merged mask here

        boxes = pred_dict.get('boxes', None)
        scores = pred_dict.get('scores', None)

        # If no detections -> save zero mask and (optionally) zero-latents
        if boxes is None or (isinstance(boxes, torch.Tensor) and boxes.numel() == 0) or (isinstance(boxes, list) and len(boxes) == 0):
            mask_np = np.zeros((orig_h, orig_w), dtype=np.uint8)
            Image.fromarray(mask_np).save(save_path)

            # If detheat requested, write zero map for completeness
            if 'detheat' in latent_flags:
                os.makedirs(latent_det_dir, exist_ok=True)
                tiff.imwrite(os.path.join(latent_det_dir, base_stem + "_detheat.tif"),
                             mask_np.astype(np.float32))
            progress_bar.update()
            continue

        if isinstance(boxes, list):
            boxes = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes = boxes.to(dtype=torch.float32)

        # Set image in SAM and map boxes to its internal transform
        sam_predictor.set_image(orig_image)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes, (orig_h, orig_w)
        ).to(sam_predictor.model.device)

        # Predict SAM masks per box (expecting low_res logits as 3rd return)
        # Some forks return (masks, ious, low_res). Be defensive.
        low_res = None
        try:
            masks, ious, low_res = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=args.sam_multimask
            )
        except TypeError:
            # Fallback signature without low_res in rare forks
            masks, ious = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=args.sam_multimask
            )

        pred_dict['masks'] = masks  # for overlay

        # Save per-instance masks then merge
        save_prefix = os.path.join(out_dir, base_stem)
        save_instance_masks(masks, save_prefix)

        # Save overlay if requested
        if args.save_overlay:
            overlay_dir = os.path.join(out_dir, "overlay")
            os.makedirs(overlay_dir, exist_ok=True)
            overlay_path = os.path.join(overlay_dir, base_name)
            draw_and_save(orig_image, pred_dict, overlay_path, random_color=True, show_label=True)

        # ------- Export Latents -------
        # 1) soft (SAM low-res logits -> sigmoid -> upsample)
        if 'soft' in latent_flags:
            os.makedirs(latent_soft_dir, exist_ok=True)
            soft_ok = export_soft_latent(low_res, orig_h, orig_w,
                                         os.path.join(latent_soft_dir, base_stem + "_soft.tif"))
            if not soft_ok:
                # Graceful fallback: use binary masks merged and lightly blurred as a proxy
                m = masks.detach().float()
                m = m.squeeze(1) if m.ndim == 4 and m.shape[1] == 1 else m  # (N,H,W)
                if m.ndim == 3:
                    m_merged = torch.amax(m, dim=0, keepdim=True).unsqueeze(0)  # (1,1,H,W)
                    m_lr = F.interpolate(m_merged, size=(orig_h, orig_w), mode='nearest')  # keep shape
                    arr = m_lr.squeeze().detach().cpu().numpy().astype(np.float32)
                    arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=1.2, sigmaY=1.2)
                    tiff.imwrite(os.path.join(latent_soft_dir, base_stem + "_soft.tif"), norm01(arr).astype(np.float32))

        # 2) embednorm and 3) embedK
        if ('embednorm' in latent_flags) or ('embedk' in latent_flags):
            img_embed = get_sam_embedding(sam_predictor)  # (1,C,H',W') or None
            if isinstance(img_embed, torch.Tensor):
                export_embednorm_and_embedK(
                    img_embed, orig_h, orig_w,
                    latent_embed_root, base_stem,
                    K=args.embed_k if ('embedk' in latent_flags) else 0
                )

        # 4) detheat (detector heatmap)
        if 'detheat' in latent_flags:
            os.makedirs(latent_det_dir, exist_ok=True)
            export_det_heatmap(boxes, scores, orig_h, orig_w,
                               os.path.join(latent_det_dir, base_stem + "_detheat.tif"))

        progress_bar.update()

    # Merge instance masks into final PNG per image
    image_mask_pair = make_pair(out_dir)
    mask_add(image_mask_pair)


if __name__ == '__main__':
    main()
