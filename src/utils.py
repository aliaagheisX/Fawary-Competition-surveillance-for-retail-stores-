import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from constants import TRAIN_PATH, TEST_PATH

import cv2
import torchvision
import numpy as np
from PIL import Image

num_seq_keys = ["frameRate", "seqLength", "imWidth", "imHeight"]
def get_info_from_seqinfo(vid = "02", path = None) -> dict:
    """ read sequence info and return object """
    if path == None:  path = TRAIN_PATH/f"{vid}/seqinfo.ini"
    
    with open(path, 'r') as f:
        lines = f.readlines()[1:-1] 

    seqinfo = {}
    for l in lines:
        [k, v] = l.strip().split("=")
        seqinfo[k] = int(v) if k in num_seq_keys else v
        
    return seqinfo

def get_frame(vid = "02", fnum = 1) -> Image:
    """ return frame in train set """
    frame_name = str(fnum).zfill(6)   
    frame_path = TRAIN_PATH / f"{vid}/img1/{frame_name}.jpg"
    return Image.open(frame_path)

def get_frame_test(fnum = 1) -> Image:
    """ return frame in test set """
    frame_name = str(fnum).zfill(6)   
    frame_path = TEST_PATH / f"img1/{frame_name}.jpg"
    return Image.open(frame_path)


def show_gt_frame(vid="02", fnum=1, df=None) -> np.ndarray:
    """ read from train set and draw bound boxes """
    frame = get_frame(vid, fnum)
    frame = np.array(frame)  # Convert PIL Image to NumPy array if needed
    
    boxes = df.query(f"vid == '{vid}' & fnum == {fnum}")[['x', 'y', 'w', 'h', 'class']]
    
    for _, row in boxes.iterrows():
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        class_id = int(row['class'])

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put text with background for better visibility
        text = f"class {class_id}"
        
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x, text_y = x, y - 5
        cv2.rectangle(frame, (text_x, text_y - text_size[1] - 3), (text_x + text_size[0] + 3, text_y), (0, 255, 0), -1)
        cv2.putText(frame, text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return frame


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x) # C, H, W
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255 # H, W, C 
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


if __name__ == "__main__":
    get_frame().show()
    get_frame_test().show()
