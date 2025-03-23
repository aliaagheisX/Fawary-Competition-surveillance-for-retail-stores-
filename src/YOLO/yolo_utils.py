import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from constants import TRAIN_PATH, TEST_PATH, DATA_DIR
from dataset import df_train, df_test
from utils import get_info_from_seqinfo

import shutil
from tqdm.auto import tqdm
import albumentations as A
from PIL import Image
import numpy as np

YOLO_IMGS = DATA_DIR / "yolo/images"
YOLO_LABELS =  DATA_DIR / "yolo/labels"

YOLO_IMGS.mkdir(exist_ok=True, parents=True)
YOLO_LABELS.mkdir(exist_ok=True, parents=True)

(YOLO_IMGS / "train").mkdir(exist_ok=True, parents=True)
(YOLO_IMGS / "val").mkdir(exist_ok=True, parents=True)
(YOLO_LABELS / "train").mkdir(exist_ok=True, parents=True)
(YOLO_LABELS / "val").mkdir(exist_ok=True, parents=True)


transform  = A.Compose([
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
        A.RandomFog(p=0.1),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=[], clip=True, min_visibility=0.7))
        

def mot_to_yolo_format():
    img_id = 0
    # training
    for seq in ["02", "03", "05"]:
        seq_info = get_info_from_seqinfo(seq)
        width, height = seq_info['imWidth'], seq_info['imHeight']
        
        for i in tqdm(range(1, 1000 + 1, 5)): 
            img_id += 1
            
            frame_name = str(i).zfill(6)               
            img_path = TRAIN_PATH / f"{seq}/img1/{frame_name}.jpg"
            fnum = int(img_path.name[:-4])
            bboxes = df_train.query(f"fnum == {fnum} and vid == '{seq}' and (`class` == 1 | `class` == 7)")[['x', 'y', 'w', 'h']].values.tolist()
            bboxes = [[(x+w/2)/width, (y+h/2)/height, w/width, h/height] for [x, y, w, h] in bboxes]
            
            
            img = Image.open(img_path)
            transformed = transform(image = np.array(img), bboxes=bboxes)
            
            
            # shutil.copy(img_path, YOLO_IMGS / "train" / f"{img_id}.jpg")
            Image.fromarray(transformed['image']).save(YOLO_IMGS / "train" / f"{img_id}.jpg")
            
            
            bboxes_txt = [f"{0} {xc} {yc} {w} {h}" for [xc, yc, w, h] in transformed['bboxes']]
            
            with open(YOLO_LABELS / "train" / f"{img_id}.txt", 'w') as f:
                f.write("\n".join(bboxes_txt))
    # TEST
    for img_path in tqdm((DATA_DIR / "test").rglob("*.jpg")): 
        img_id += 1
        shutil.copy(img_path, YOLO_IMGS / "val" / f"{img_id}.jpg")
        
        fnum = int(img_path.name[:-4])
        bboxes = df_test.query(f"fnum == {fnum} and (`class` == 1 | `class` == 7)")[['x', 'y', 'w', 'h']].values.tolist()
        bboxes = [[(x+w/2)/width, (y+h/2)/height, w/width, h/height] for [x, y, w, h] in bboxes]
        
        bboxes_txt = [f"{0} {xc} {yc} {w} {h}" for [xc, yc, w, h] in bboxes]
        
        with open(YOLO_LABELS / "val" / f"{img_id}.txt", 'w') as f:
            f.write("\n".join(bboxes_txt))

            

if __name__ == "__main__":
    # print(df_test.head())
    mot_to_yolo_format()