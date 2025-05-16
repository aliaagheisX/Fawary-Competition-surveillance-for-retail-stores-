import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ----------- local imports ----------- 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import albumentations as A
from torchvision import transforms
from dataset import df_train, df_test
from utils import get_frame, get_frame_test
import numpy as np
import cv2
class CustomDataset(Dataset):
    def __init__(self, df=df_train, is_validation=False):
        super().__init__()
        self.is_validation = is_validation
        self.transform  = A.Compose([
            A.HorizontalFlip(p=0.5),
            # Replace ShiftScaleRotate with Affine
            A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), p=0.5),
            
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=0.5),
            
            # Noise/quality variations
            A.GaussNoise(p=0.3),
            A.ISONoise(p=0.3),
            A.MotionBlur(p=0.3),
            # Occlusion simulation
            A.CoarseDropout(p=0.3),
            # Weather effects (use sparingly)
            A.RandomFog(p=0.2),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[], clip=True, min_visibility=0.7))
        
        self.to_tensor = transforms.ToTensor()
        
        if is_validation:
            self.df = df_test.copy()
            self.group_by = ['fnum']
        else:
            self.df = df.copy()
            self.group_by = ['vid', 'fnum']
        
        # =================== preprocessing ========================== 
        self.df['x1'] = self.df['x'] + self.df['w']
        self.df['y1'] = self.df['y'] + self.df['h']
        
        self.df['bbox_info'] = self.df[['x', 'y', 'x1', 'y1']].values.tolist()
        
        # Filter based on validation flag
        base_query = '(`class` == 1 | `class` == 7) and fnum <= 1000 and conf == 1'

        self.df = (self.df
                   .query(f'{base_query}')
                  .groupby(self.group_by)['bbox_info']
                  .apply(list)
                  .reset_index()
                )
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        
        if self.is_validation:
            img = get_frame_test(row['fnum'])
            transformed = {'image': np.array(img), 'bboxes': row['bbox_info']}
        else:
            img = get_frame(row['vid'], row['fnum'])
            transformed = self.transform(image = np.array(img), bboxes=row['bbox_info'])
        
        # scale image by 2
        # img = cv2.resize(transformed['image'], (img.width * 2, img.height * 2), Image.BILINEAR)
        # bboxes = 2*np.array(transformed['bboxes'])
        img = transformed['image']
        bboxes = np.array(transformed['bboxes'])
        
        # output targets
        targets = {
            'boxes': torch.from_numpy(bboxes).type(torch.float32),
            'labels': torch.ones(len(bboxes), dtype=torch.int64),
        }

        return self.to_tensor(img), targets
    
    def __len__(self):
        return len(self.df)
    

def get_train_val_datasets():
    train_dataset = CustomDataset(is_validation=False)
    val_dataset = CustomDataset(is_validation=True)
    return train_dataset, val_dataset

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

if __name__ == "__main__":
    batch_size = 16
    train_dataset, val_dataset = get_train_val_datasets()
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Test a batch
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    print("\nSample train batch:", train_batch[0][0].shape)
    print("\nSample val batch:", val_batch[0][0].shape)
    
