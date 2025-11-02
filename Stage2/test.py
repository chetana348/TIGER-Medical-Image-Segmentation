import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import Data_Gen
from model.network import * #DATTNet
from utils import *  # Assuming it's defined in utils

# --- Paths ---
test_data_path = '/users/PAS3110/sephora20/workspace/PDAC/data/pros/test/images_soft/'
test_label_path = '/users/PAS3110/sephora20/workspace/PDAC/data/pros/test/labels/'
test_prompt_file="/users/PAS3110/sephora20/workspace/PDAC/data/pros/test/prompts.json"

model_path = '/users/PAS3110/sephora20/workspace/PDAC/Tiger/Stage2/outputs/pros/single_stage1_prompt/soft_latent/weights/best_model.pth'
save_dir = '/users/PAS3110/sephora20/workspace/PDAC/Tiger/Stage2/outputs/pros/single_stage1_prompt/soft_latent/pred/'
os.makedirs(save_dir, exist_ok=True)

# --- Load Dataset and Model ---
test_dataset = Data_Gen(test_data_path, test_label_path, test_prompt_file)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)
print(f"Total test images: {len(test_dataset)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = Net(in_channels = 2, out_channels=2).to(device)
network.load_state_dict(torch.load(model_path, map_location=device))
network.eval()

dice_fn = DiceScore(num_classes=2).to(device)
iou_fn = IoU(num_classes=2).to(device)

# --- Metrics ---
dice_total = 0.0
iou_total = 0.0
ravd_total = 0.0
num_batches = 0
num_images = 0

dice_list = []   # NEW
iou_list = []    # NEW
ravd_list = []

# --- Evaluation Loop ---
with torch.no_grad():
    for batch_idx, (images, labels, texts, stems, image_filenames) in tqdm(
        enumerate(test_dataloader), total=len(test_dataloader), desc="Predicting"):

        images = images.to(device)
        labels = labels.to(device)
        texts  = list(texts)
        outputs = network(images, texts=texts)[0]  # If model returns tuple

        # Dice Score
        dice = dice_fn(outputs, labels)
        dice_value = dice.item()
        dice_total += dice_value
        dice_list.append(dice_value)   # NEW

        # IoU Score
        iou = iou_fn(outputs, labels)
        iou_value = iou.item()
        iou_total += iou_value
        iou_list.append(iou_value)     # NEW

        num_batches += 1

        # Get predicted mask for RAVD and saving
        preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)  # [B, H, W]

        # Compute RAVD (using numpy)
        ravd_vals = ravd_batch(labels.cpu().numpy(), preds.cpu().numpy())
        ravd_total += np.sum(ravd_vals)
        num_images += preds.size(0)
        ravd_list.extend(ravd_vals)   # NEW

        # Save predicted masks
        preds_np = preds.cpu().numpy()
        for i in range(preds_np.shape[0]):
            pred_image = (preds_np[i].astype(np.uint8)) * 255
            filename = os.path.basename(image_filenames[i])
            save_path = os.path.join(save_dir, filename)
            Image.fromarray(pred_image).save(save_path)

# --- Final Results ---
average_dice = dice_total / num_batches
average_iou = iou_total / num_batches
average_ravd = ravd_total / num_images

# NEW: Compute std dev over batches
dice_std = np.std(dice_list)
iou_std = np.std(iou_list)
ravd_std = np.std(ravd_list)

print(f"\n Average Dice coefficient: {average_dice:.4f} ± {dice_std:.4f}")
print(f" Average IoU: {average_iou:.4f} ± {iou_std:.4f}")
print(f" Average RAVD (Relative Absolute Volume Difference): {average_ravd:.4f} ± {ravd_std:.4f}")
