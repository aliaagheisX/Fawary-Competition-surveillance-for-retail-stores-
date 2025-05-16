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
from torchvision import transforms
# ----------- model imports ----------- 
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

def inference(model, images, device):
    model.eval()
    to_tensor = transforms.ToTensor()
    
    if isinstance(images, list):
        image_tensor = [to_tensor(image).to(device) for image in images]
    else:        
        image_tensor = [to_tensor(images).to(device)]

    with torch.no_grad():
        results = model(image_tensor)

    return results    
        
def draw_rcnn_results(results, images):
    result_images = []
    categories = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]

    for image in images:
        image_np = np.array(image)
        # Draw bounding boxes
        for score, label, box in zip(results[0]["scores"], results[0]["labels"], results[0]["boxes"]):
            label_text = f"{categories[label.item()]}: {round(score.item(), 2)}"
            if "person" not in label_text: continue 
            if score < 0.5: continue
                
            box = [round(i, 2) for i in box.tolist()]
            x1, y1, x2, y2 = map(int, box)
            
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)        
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text_x, text_y = x1, y1 - 5
            cv2.rectangle(image_np, (text_x, text_y - text_size[1] - 3), (text_x + text_size[0] + 3, text_y), (0, 255, 0), -1)
            cv2.putText(image_np, label_text, (text_x, text_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        result_images.append(Image.fromarray(image_np))
        
    return result_images

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    images = [get_frame("02", 1), get_frame("02", 2)]
    
    print(f"Start init Models on {device}..")
    model  = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)

    print("Start inference ....")
    results = inference(model, images, device)

    print("Start Drawing ....")
    result_images = draw_rcnn_results(results, images)
    
    result_images[0].show()
    result_images[1].show()
    
if __name__ == "__main__":
    main()