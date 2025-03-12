import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from constants import TRAIN_PATH, TEST_PATH
from utils import get_info_from_seqinfo, get_frame, get_frame_test

from FasterRcnn.tracking import *


from tqdm.auto import tqdm
import supervision as sv
import numpy as np
import pandas as pd

# Fnum, tracker_id, x, y, w, h, confidence, -1, -1, -1
def predict_video(get_detections_fn, th=0.5):
    seqlen = len(list(TEST_PATH.rglob("*.jpg")))
    
    data = []
    for frame_num in tqdm(range(1, seqlen + 1)):
        frame = get_frame_test(frame_num)
        
        detections = get_detections_fn(frame, th)
        
        for i in range(len(detections.tracker_id)):
            data.append([
                frame_num, 
                int(detections.tracker_id[i]), 
                float(detections.xyxy[i, 0]), 
                float(detections.xyxy[i, 1]), 
                float(detections.xyxy[i, 2] - detections.xyxy[i, 0]), 
                float(detections.xyxy[i, 3] - detections.xyxy[i, 1]), 
                float(detections.confidence[i]), 
                int(detections.class_id[i])
            ])
        
    columns = ["fnum", "id", "x", "y", "w", "h", "cf_score", "class"]
    return pd.DataFrame(data, columns=columns)


if __name__ == "__main__":
    get_detections_fn = get_rcnn_detections_fn()
    df = predict_video(get_detections_fn, th=0.7)
    
    df.to_csv("fasterrcnn_batch8_freeze_test_th07.csv", header=None, index=None)
    