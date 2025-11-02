import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
from tqdm import tqdm
import argparse
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle
from sklearn.model_selection import train_test_split
import tifffile as tiff
from skimage import transform as tr #, img_as_ubyte
from dataset import *
from model.network import *
from utils import *

train_data_path = '/users/PAS3110/sephora20/workspace/PDAC/data/pros/train/images_soft/'
train_label_path = '/users/PAS3110/sephora20/workspace/PDAC/data/pros/train/labels/'
train_prompt_file="/users/PAS3110/sephora20/workspace/PDAC/data/pros/train/prompts.json"

test_data_path = '/users/PAS3110/sephora20/workspace/PDAC/data/pros/val/images_soft/'
test_label_path = '/users/PAS3110/sephora20/workspace/PDAC/data/pros/val/labels/'
test_prompt_file="/users/PAS3110/sephora20/workspace/PDAC/data/pros/val/prompts.json"

dataset = Data_Gen(train_data_path, train_label_path, train_prompt_file, mode = 'train')
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

test_dataset = Data_Gen(test_data_path, test_label_path, test_prompt_file, mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

network = Net(in_channels = 2, out_channels=2)   
network = network.to('cuda')

criterion = DiceLoss(num_classes=2)
accuracy_metric = DiceScore(num_classes=2)
iou_metric = IoU(num_classes=2)

optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, betas=(0.9,0.999))

# Define the directory to save models and TensorBoard logs
save_dir = '/users/PAS3110/sephora20/workspace/PDAC/Tiger/Stage2/outputs/pros/single_stage1_prompt/soft_latent/weights/'
os.makedirs(save_dir, exist_ok=True)


# Training loop
num_epochs = 1000
best_accuracy = 0.0

for epoch in range(num_epochs):
    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0  # Initialize IoU accumulation
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch_idx, (images, labels, texts, stems, img_name) in progress_bar:
        # Move images and labels to GPU
        images = images.cuda()
        labels = labels.cuda()
        texts  = list(texts)
        #print(texts[0])
        # Forward pass
        optimizer.zero_grad()
        outputs = network(images, texts=texts)
        outputs = outputs[0]
        #outputs = F.interpolate(outputs, size=(128, 128), mode='bilinear', align_corners=True)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Calculate accuracy (Dice score)
        accuracy = accuracy_metric(outputs, labels)
        total_accuracy += accuracy.item()

        # Calculate IoU
        iou = iou_metric(outputs, labels)
        total_iou += iou.item()

        progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), iou=iou.item())  # Update progress bar with current loss, accuracy, and IoU

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    avg_iou = total_iou / len(dataloader)  # Average IoU for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, IoU: {avg_iou:.4f}')
    
    # Save model for the epoch
   # model_name = f'your_model_{epoch+1}.pth'
    #model_path = os.path.join(save_dir, model_name)
    #torch.save(network.state_dict(), model_path)
    #print(f"Model saved for epoch {epoch+1} as {model_path}")

    # Validation loop to calculate Dice coefficient and IoU
    network.eval()
    total_dice = 0.0
    total_val_loss = 0.0
    total_val_iou = 0.0  # Initialize IoU accumulation for validation
    progress_bar_val = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Validation", leave=False)
    with torch.no_grad():
        for batch_idx, (images_test, labels_test, texts, stems, img_name) in progress_bar_val:
            images_test = images_test.cuda()
            labels_test = labels_test.cuda()
            texts  = list(texts)
            outputs_test = network(images_test, texts=texts)
            outputs_test = outputs_test[0]
            #outputs_test= F.interpolate(outputs_test, size=(128, 128), mode='bilinear', align_corners=True)
            val_loss = criterion(outputs_test, labels_test)
            dice = accuracy_metric(outputs_test, labels_test)
            iou = iou_metric(outputs_test, labels_test)
            total_dice += dice.item()
            total_val_loss += val_loss.item()
            total_val_iou += iou.item()
            progress_bar_val.set_postfix(loss=val_loss.item(), dice=dice.item(), iou=iou.item())  # Update progress bar with current loss, dice, and IoU

    avg_dice = total_dice / len(test_dataloader)
    avg_val_loss = total_val_loss / len(test_dataloader)
    avg_val_iou = total_val_iou / len(test_dataloader)  # Average IoU for validation
    print(f'Average Dice coefficient: {avg_dice:.4f}')
    print(f'Validation Loss: {avg_val_loss:.4f}')
    print(f'Validation IoU: {avg_val_iou:.4f}')


    # Check if current accuracy is the best so far
    if avg_dice > best_accuracy:
        best_accuracy = avg_dice
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(network.state_dict(), best_model_path)
        print(f"New best model saved with accuracy {avg_dice:.4f}")

    network.train()

print("Training finished.")






