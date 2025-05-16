import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from constants import TRAIN_PATH, TEST_PATH, DATA_DIR

import pandas as pd


get_vid_img_path = lambda vid: TRAIN_PATH/f"{vid}/img1/"
get_vid_gt_path = lambda vid: TRAIN_PATH/f"{vid}/gt/"

gt_v02 = pd.read_csv(f"{TRAIN_PATH}/02/gt/gt.txt", header=None)
gt_v03 = pd.read_csv(f"{TRAIN_PATH}/03/gt/gt.txt", header=None)
gt_v05 = pd.read_csv(f"{TRAIN_PATH}/05/gt/gt.txt", header=None)

gt_v02["vid"] = "02"
gt_v03["vid"] = "03"
gt_v05["vid"] = "05"

df_train = pd.concat([gt_v02, gt_v03, gt_v05])
df_train.columns = ["fnum", "id", "x", "y", "w", "h", "conf", "class", "visibility", "vid"]


df_test = pd.read_csv(DATA_DIR / "test_gt.txt", header=None)
df_test.columns = ["fnum", "id", "x", "y", "w", "h", "conf", "class", "visibility"]
