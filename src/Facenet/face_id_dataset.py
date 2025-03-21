import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ----------- local imports ----------- 
from constants import DATA_DIR, FACE_ID_TRAIN_PATH

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from deepface.modules import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit


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


def get_embedding_from_path(embedding_path = "embeddings/train_embeddings.csv"):
    """ preprocess train embedding dataframe """
    # preprocess train embedding dataframe
    df_train_embeddings = pd.read_csv(embedding_path)
    # fix embeddings to np array
    df_train_embeddings['embeddings'] = df_train_embeddings['embeddings'].apply(lambda x: np.array(x[1:-1].split(), dtype=np.float32))
    # add person as column
    df_train_embeddings['person'] = df_train_embeddings['identity'].apply(lambda p: Path(p).parent.name)
    # add image name as column `img`
    df_train_embeddings['img'] = df_train_embeddings['identity'].apply(lambda p: Path(p).name)
    # drop whole path `identity` column
    df_train_embeddings = df_train_embeddings.drop(['identity'], axis=1)

    return df_train_embeddings

def get_test_embedding_from_path(embedding_path = "embeddings/test_embeddings.csv"):
    """ preprocess train embedding dataframe """
    # preprocess train embedding dataframe
    df_test_embeddings = pd.read_csv(embedding_path)
    # fix embeddings to np array
    df_test_embeddings['embeddings'] = df_test_embeddings['embeddings'].apply(lambda x: np.array(x[1:-1].split(), dtype=np.float32))
    df_test_embeddings['img'] = df_test_embeddings['identity'].apply(lambda p: Path(p).name)
    return df_test_embeddings


def get_train_val_set(df_train_embeddings, seed = 1710, unknown_cnt = 5):
    df = df_train_embeddings.copy()
    # =============================
    # by random select 5 people to go to validation set
    # =============================
    # 1. get random people
    np.random.seed(seed)
    rand_people_index = np.random.randint(0, 124 + 1, unknown_cnt)

    rand_peple_mask = df['person'].apply(lambda x: int(x.split('_')[1]) in rand_people_index)
    # make train set without it
    train_df = df[~rand_peple_mask].copy()
    # make validation set with it
    val_df =  df[rand_peple_mask].copy()
    val_df['gt'] = "doesn't_exist"
    # =============================
    # foreach person sample percentage(0.2) of images for validation set
    # =============================
    X = train_df.index
    y = train_df['person']

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, val_idx in sss.split(X, y):
        v = train_df.iloc[val_idx].copy()
        v['gt'] = v['person']
        val_df = pd.concat([val_df, v])
        #
        train_df = train_df.iloc[train_idx].copy()
        
    return train_df, val_df


