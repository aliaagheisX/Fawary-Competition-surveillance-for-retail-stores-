import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import MODEL_DIR, DATA_DIR

import torch
from ultralytics import YOLO # yolo model
# shut down loggings
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = YOLO(MODEL_DIR / "best(8).pt").to(device)
    
    model.train(data=DATA_DIR / 'custom_dataset.yaml',  
                epochs=100,                 # Total epochs
                project='fawary_retail_store', 
                name='run_mot',
                imgsz=640,                  
                batch=16,                    
                augment=True,
                lr0=1e-4,
                save_period=10,             # Save model every 10 epochs
                device=0,                   # GPU index
                resume=False,               # Ensure fresh training
                val=True,                   # Run validation
                verbose=True,                # Detailed logs
                rect=True
    )
    
if __name__ == "__main__":
    train()