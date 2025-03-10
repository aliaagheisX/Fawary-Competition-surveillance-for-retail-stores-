import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import pandas as pd


get_vid_img_path = lambda vid: TRAIN_PATH/f"{vid}/img1/"
get_vid_gt_path = lambda vid: TRAIN_PATH/f"{vid}/gt/"