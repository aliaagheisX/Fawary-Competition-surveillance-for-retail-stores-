import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastcore.script import *
from tqdm.auto import tqdm
# ----------- local imports ----------- 
from utils import build_wandb_run
from constants import MODEL_DIR
from FasterRcnn.CustomDataset import CustomDataset, get_train_val_datasets

# ----------- model imports ----------- 
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device="cpu").manual_seed(42)

batch_size = 8
num_epochs = 10
log_steps = 200
accumlation_step = 2
collate_fn=lambda x: tuple(zip(*x))


run = build_wandb_run(config = {
    "dataset": "MOT20",
    "learning_rate": 0.005,
    "momentum": 0.9,
    "weight_decay": 0.0005,
    "batch_size": batch_size,
    "num_epochs": 3,
    "wandb_project": "fawry-competition",
})

train_dataset = CustomDataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=generator)

val_dataset = CustomDataset(is_validation=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)


# Load Dataset
def validation_loop(model, val_loader):
    total_loss = 0.0
    for images, targets in tqdm(val_loader, desc="Validation"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        
    avg_loss = total_loss / len(val_loader)
    print(f"VAL_LOSS: {avg_loss:.4f}")
    run.log({'val_loss': avg_loss})

def train_loop(model, scaler, optimizer, train_loader):
    train_loss = 0.0
    for i, (images, targets) in enumerate(tqdm(train_loader)):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        scaler.scale(losses).backward()
        
        if (i+1)%accumlation_step == 0 or (i+1) == len(train_loader):            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        train_loss += losses.item()
        
        # Log loss every `log_steps`
        if (i + 1) % log_steps == 0:
            avg_loss = train_loss / (i + 1)
            print(f"Step [{i+1}/{len(train_loader)}], Train Loss: {avg_loss:.4f}")
            run.log({"train_loss": avg_loss})
            
    avg_loss = train_loss / len(train_loader)
    print(f"Final Epoch Train Loss: {avg_loss:.4f}")
    run.log({"train_loss": avg_loss})
    run.log({"final_train_loss": avg_loss})

def train(train_loader, val_loader, checkpoint_path = MODEL_DIR /"fastercnn_batch_8_epoch_5.pth.tar"):
    model  = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # freeze the backbone - layer 6
    # for p in model.backbone.parameters():  p.requires_grad = False
    # for p in model.roi_heads.box_head.fc6.parameters(): p.requires_grad = False
    
    # # freeze all model    
    for p in model.parameters(): p.requires_grad = False 
    # # 1. Unfreeze later layers of the backbone (ResNet layer4)
    for p in model.backbone.body.layer4.parameters():
        p.requires_grad = True  
    # # 2. Unfreeze the fc7 layer in roi_heads (better pedestrian classification & bounding boxes)
    for p in model.roi_heads.box_head.fc7.parameters():
        p.requires_grad = True  
    for p in model.roi_heads.box_predictor.parameters():  # Classification & bbox head
        p.requires_grad = True  

    # # 3. Unfreeze the Feature Pyramid Network (FPN)   => help small object detection
    for p in model.backbone.fpn.parameters():
        p.requires_grad = True  
    # # 4. Unfreeze the Region Proposal Network (RPN) (better proposals for crowded scenes)
    for p in model.rpn.parameters():
        p.requires_grad = True  
        
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"Total trainable parameters: {sum(p.numel() for p in params)}")

    optimizer = torch.optim.SGD([
            {'params': model.backbone.body.layer4.parameters(), 'lr': 1e-4},  # Lower LR for backbone
            {'params': model.roi_heads.box_head.fc7.parameters(), 'lr': 1e-3},  # Higher LR for ROI head
            {'params': model.backbone.fpn.parameters(), 'lr': 1e-3},  # Fine-tune FPN
            {'params': model.rpn.parameters(), 'lr': 1e-3}  # Fine-tune RPN
        ], momentum=0.9, weight_decay=1e-4)


    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    scaler = torch.amp.GradScaler()


    for epoch in range(num_epochs):
        model.train()

        train_loop(model, scaler, optimizer, train_loader)
        
        validation_loop(model, val_loader)
        
        torch.save({
            'model_state_dict': model.state_dict(),
        }, MODEL_DIR / f"fastercnn_batch_{batch_size}_epoch_{epoch+1}_mot20.pth.tar")
        
        lr_scheduler.step()
        
if __name__ == "__main__":
    train(train_loader, None)
    run.finish()