import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tqdm.auto import tqdm

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ============== Models ============== 
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import supervision as sv # tracker 
# shut down loggings
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ----------- local imports ----------- 
from utils import show_images, get_frame, get_frame_test, get_info_from_seqinfo
from constants import TRAIN_PATH, TEST_PATH
from dataset import df_train, get_vid_img_path, get_vid_gt_path
from FasterRcnn.inference import *
from FasterRcnn.CustomDataset import CustomDataset, collate_fn

# ----------- model imports ----------- 
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator().manual_seed(42)
dtype = torch.bfloat16

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)), generator=generator)


model  = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).type(dtype)
# freeze the backbone (it will freeze the body and fpn params)
for p in model.backbone.parameters():  p.requires_grad = False
# freeze the fc6 layer in roi_heads
for p in model.roi_heads.box_head.fc6.parameters(): p.requires_grad = False


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for i, (images, targets) in enumerate(tqdm(dataloader)):
        images = [img.to(device).type(dtype) for img in images]

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        # calc
        with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
        losses.backward()
        optimizer.step()
        
        train_loss += losses.item()
        
    print(f"STEP: {epoch}\t train_loss: {(train_loss/len(dataloader)):.2f}")
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss/len(dataloader),
            }, "./models/fastrcnn_freeze_8batchs_3epochs.pth")    
    lr_scheduler.step()
    
