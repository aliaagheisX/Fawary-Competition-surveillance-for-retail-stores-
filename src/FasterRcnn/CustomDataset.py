import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ----------- local imports ----------- 
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from dataset import df_train
from utils import get_frame

class CustomDataset(Dataset):
    def __init__(self, df = df_train):
        super().__init__()
        
        self.to_tensor = transforms.ToTensor()
        
        self.df = df.copy()
        self.df['x1'] = self.df['x'] + self.df['w']
        self.df['y1'] = self.df['y'] + self.df['h']
        
        self.df['bbox_info'] = self.df[['x', 'y', 'x1', 'y1']].values.tolist()
        self.df = (self.df
                            .query('`class` == 1 | `class` == 7') # filter classes
                            .query('fnum <= 1000')                # make frame number as we only have samples
                            .groupby(['vid', 'fnum'])['bbox_info']# 
                            .apply(list)
                            .reset_index()
                    )
        
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = self.to_tensor( get_frame(row['vid'],row['fnum']) )
        targets = {}
        targets['boxes'] = torch.tensor(row['bbox_info'], dtype=torch.float32)
        targets['labels'] =  torch.tensor([1]*len(row['bbox_info']), dtype=torch.int64)

        return img, targets
    
    def __len__(self):
        return len(self.df)

def collate_fn(batch):
    images, targets = zip(*batch)  # Unzips batch into two tuples
    
    return list(images), list(targets)  # Convert tuples to lists

if __name__ == "__main__":
    batch_size = 16
    dataset = CustomDataset()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn )
    
    xb = next(iter(dataloader))
    
    print(xb)
    
