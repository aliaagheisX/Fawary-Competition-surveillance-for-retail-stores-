import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external/TrackEval')))
from constants import TRAIN_PATH, TEST_PATH, DATA_DIR
from utils import get_info_from_seqinfo, get_frame, get_frame_test
from dataset import get_vid_img_path, df_train
from FasterRcnn.tracking import *

from tqdm.auto import tqdm
import supervision as sv
import numpy as np
import pandas as pd
from trackeval import Evaluator, datasets, metrics


# Fnum, tracker_id, x, y, w, h, confidence, -1, -1, -1
def predict_video(get_detections_fn, vid = "02", th=0.7):
    vid_path = get_vid_img_path(vid)
    seqlen = len(list(vid_path.rglob("*.jpg")))
    
    data = []
    for frame_num in tqdm(range(1, seqlen + 1)):
        frame = get_frame(vid, frame_num)
        
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

def predict_video_test(get_detections_fn, th=0.7):
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


def prep_eval_tracker_vid(model_name, tracker_name):
    # take sample of gt
    sample_dir = Path(DATA_DIR/"my_sample/gt/MOT20-train/MOT20-02/gt")
    sample_dir.mkdir(exist_ok=True, parents=True)

    df_sample = df_train.query('vid == "02" & fnum <= 1000 ') \
            .drop(["vid"], axis=1) \
            .query("`class` == 1 | `class` == 7")
            
    df_sample['class'] = 1

    df_sample.to_csv(sample_dir/"gt.txt", header=None, index=None)
    # ======================================================================
    tracker_dir = Path(DATA_DIR/f"my_sample/trackers/MOT20-train/{tracker_name}/data")
    tracker_dir.mkdir(exist_ok=True, parents=True)

    df_tracker = pd.read_csv(f"src/FasterRcnn/outputs/{model_name}.csv")#.drop(['Unnamed: 0'], axis=1)
    df_tracker.to_csv(tracker_dir / 'MOT20-02.txt', header=None, index=None)

def prep_eval_tracker_test(model_name, tracker_name):
    # take sample of gt
    sample_dir = Path(DATA_DIR/"my_sample/gt/MOT20-train/MOT20-01/gt")
    sample_dir.mkdir(exist_ok=True, parents=True)

    test_path_gt = DATA_DIR / "test_gt.txt"
    df = pd.read_csv(test_path_gt, header=None, )

    df = df[(df[7] == 1) | (df[7] == 7)]
    df[7] = 1

    df.to_csv(sample_dir/"gt.txt", header=None, index=None)   
    # ======================================================================
    tracker_dir = Path(f"./datasets/my_sample/trackers/MOT20-train/{tracker_name}/data")
    tracker_dir.mkdir(exist_ok=True, parents=True)

    df_tracker = pd.read_csv(f"src/FasterRcnn/outputs/{model_name}_test.csv")
    df_tracker.to_csv(tracker_dir / 'MOT20-01.txt', header=None, index=None)

if __name__ == "__main__":
    tracker_name = "F"
    model_name = "fastercnn_batch_8_epoch_5_mot20"
    get_detections_fn = get_rcnn_detections_fn(model_name = model_name+".pth.tar")
    
    df = predict_video(get_detections_fn)
    df.to_csv(f"src/FasterRcnn/outputs/{model_name}.csv", header=None, index=None)
    
    df = predict_video_test(get_detections_fn)
    df.to_csv(f"src/FasterRcnn/outputs/{model_name}_test.csv", header=None, index=None)
    
    prep_eval_tracker_vid(model_name, tracker_name)
    prep_eval_tracker_test(model_name, tracker_name)
    
    dataset_config = {
        "GT_FOLDER": str(DATA_DIR / "my_sample/gt"),  
        "TRACKERS_FOLDER": str(DATA_DIR / "my_sample/trackers"),
        "TRACKERS_TO_EVAL": tracker_name,
        "SEQMAP_FILE": str(DATA_DIR / "my_sample/gt/seqmaps/MOT20-train.txt"), 
        "BENCHMARK": "MOT20", 
    }

    dataset = datasets.MotChallenge2DBox(dataset_config)
    metric = metrics.HOTA()
    evaluator = Evaluator()

    # Run evaluation
    result = evaluator.evaluate([dataset], [metric])