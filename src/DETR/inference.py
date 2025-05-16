import os
import sys
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ----------- local imports ----------- 
from utils import get_frame
from dataset import get_vid_img_path, get_vid_gt_path
# ----------- imports ----------- 
import cv2
import torch
import numpy as np
from PIL import Image
# ----------- model imports ----------- 
from transformers import DetrImageProcessor, DetrForObjectDetection


# ==================================== inference ====================================
def inference(model, processor, image: Image, threshold = 0.2):
    model.eval()

    inputs = processor(images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # post process boxes
    results = processor.post_process_object_detection(outputs, target_sizes=torch.tensor([[image.height, image.width]]), threshold=threshold)[0]
    return results

def draw_detr_results(results, image, model):
    image_np = np.array(image)
    # Draw bounding boxes
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
        if "person" not in label_text: continue 
            
        box = [round(i, 2) for i in box.tolist()]
        x1, y1, x2, y2 = map(int, box)
        
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)        
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x, text_y = x1, y1 - 5
        cv2.rectangle(image_np, (text_x, text_y - text_size[1] - 3), (text_x + text_size[0] + 3, text_y), (0, 255, 0), -1)
        cv2.putText(image_np, label_text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return Image.fromarray(image_np)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = get_frame("02", 1)
    
    print(f"Start init Models on {device}..")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm").to(device)

    print("Start inference ....")
    results = inference(model, processor, image)

    print("Start Drawing ....")
    draw_detr_results(results, image, model).show()
    
if __name__ == "__main__":
    main()