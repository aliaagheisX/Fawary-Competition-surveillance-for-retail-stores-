import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ----------- local imports ----------- 
from constants import DATA_DIR, FACE_ID_TRAIN_PATH

import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from deepface.modules import preprocessing


def load_faces_in_batch(batch_size=16, image_dir = FACE_ID_TRAIN_PATH):
    """ load images in batchs """
    paths = []
    images = []
    imgs_path = sorted(list(image_dir.rglob("*.jpg")))
    total_len = len(imgs_path)
    
    for i, img_path in enumerate(tqdm(imgs_path)):
        # ============= preprocess =============
        img = np.array( Image.open(img_path) )
        img = preprocessing.resize_image(
            img=img,
            target_size=(160, 160),
        )
        img = preprocessing.normalize_input(img=img, normalization="base")
        
        # ============= add image =============
        paths.append(img_path)
        images.append(img)
        
        # ============= yield batch =============
        if (i + 1) % batch_size == 0 or (i+1) == total_len:
            yield (paths, np.array(images)[:,0,:,:])
            images = []
            paths = []
            
            
def make_small_dataset(
        size = (160, 160),
        train_imgs_path = FACE_ID_TRAIN_PATH, 
        train_small_path: Path = DATA_DIR/"face_identification/train_small"
    ):
    """ copy & resize each image in train_imgs_path  to  train_small_path """
    
    train_small_path.mkdir(exist_ok=True, parents=True)
    
    for img_path in tqdm(train_imgs_path.rglob("*.jpg")):
        person_path = train_small_path / img_path.parent.name
        person_path.mkdir(exist_ok=True, parents=True)
        
        new_img_path = person_path / img_path.name
        
        Image.open(img_path).resize(size, Image.LANCZOS).save(new_img_path)
