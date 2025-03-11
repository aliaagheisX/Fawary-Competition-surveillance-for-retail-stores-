import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ----------- local imports ----------- 
from utils import get_frame
from dataset import get_vid_img_path, get_vid_gt_path

from FasterRcnn.inference import *  

import supervision as sv # tracker 
from constants import MODEL_DIR

def get_detections_from_rcnn_results(results):
    boxes = results[0]["boxes"].cpu().numpy()
    labels = results[0]["labels"].cpu().numpy()
    confidence = results[0]["scores"].cpu().numpy()

    mask = (labels == 1) & (confidence > 0.5)
    boxes = boxes[mask]
    labels = labels[mask]
    confidence = confidence[mask]

    return sv.Detections(
        xyxy=boxes, confidence=confidence, class_id=labels
    )

def get_rcnn_detections_fn():
    """ return function pass image to get  """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    checkpoint = torch.load(MODEL_DIR / "fastrcnn_freeze_8batchs_3epochs.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    tracker = sv.ByteTrack() # to track IDs
    

    def get_image_detection(image):
        results = inference(model, image, device)
        detections = get_detections_from_rcnn_results(results)
        detections = tracker.update_with_detections(detections)
        return detections
    
    return get_image_detection

def draw_rcnn_detections(detections, image):
    box_annotator = sv.BoxAnnotator() # to draw boxes
    label_annotator = sv.LabelAnnotator() # to draw ids on boxes    
    
    categories = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
    labels = [
        f"#{tracker_id} {categories[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]
    
    annotated_frame = box_annotator.annotate(image.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
    
    return annotated_frame



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = get_frame("02", 1)


    model  = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    tracker = sv.ByteTrack() # to track IDs

    
    results = inference(model, image, device)
    detections = get_detections_from_rcnn_results(results)
    detections = tracker.update_with_detections(detections)
    
    draw_rcnn_detections(detections, image).show()
    
    

if __name__ == "__main__":
    main()