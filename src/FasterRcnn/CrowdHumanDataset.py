import json
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)
device

class CrowdHumanDataset(Dataset):
    def __init__(self, annot, IMG_PATH):
        self.annot = annot
        self.transform = transforms.ToTensor()
        self.IMG_PATH = IMG_PATH
        
    def __getitem__(self, index):
        img_id = self.annot[index]['ID']
        img = self.transform(Image.open(self.IMG_PATH / f"{img_id}.jpg"))
        
        bboxes = [annot['fbox'] for annot in self.annot[index]['gtboxes']]
        bboxes = [[x, y, x+w, y+h] for [x, y, w, h] in bboxes]
        

        targets = {}
        targets['boxes'] = torch.tensor(bboxes, dtype=torch.float32)
        targets['labels'] = torch.tensor([1]*len(targets['boxes']), dtype=torch.int64)

        return img, targets
        
    def __len__(self):
        return len(self.annot)