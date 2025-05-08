import os
import torch
import wandb
import copy
from tqdm import tqdm, trange
from torch.optim import AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.data_loader import loader
import torch.nn.functional as F
from augmentation.Augmentation import Cutout
from wandb_init import parser_init, wandb_init
from utils.metrics import calculate_metrics
from models.Model import model_dice_bce
from utils.Loss import DINOLoss
from torch.nn.utils import clip_grad_norm_
from torch import nn 
import matplotlib.pyplot as plt
import numpy as np


def get_teacher_momentum(current_epoch, max_epochs, base_m=0.996, final_m=1.0):
    # Linear momentum schedule
    return base_m + (final_m - base_m) * (current_epoch / max_epochs)

def get_teacher_temp(epoch, warmup_epochs=30, final_temp=0.07):
    start_temp = 0.04
    if epoch < warmup_epochs:
        return start_temp + (final_temp - start_temp) * epoch / warmup_epochs
    else:
        return final_temp
    
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        return self.mlp(x)


def featuremap_to_heatmap(tensor):
    """Convert a 2D tensor to a colored heatmap suitable for wandb.Image."""
    heatmap = tensor.detach().cpu()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-5)  # Normalize to [0, 1]
    heatmap_np = heatmap.numpy()

    # Apply colormap
    cmap = plt.get_cmap("viridis")  # Choose any matplotlib colormap: 'jet', 'plasma', etc.
    colored_heatmap = cmap(heatmap_np)[:, :, :3]  # Drop alpha channel

    # Convert to 8-bit RGB
    colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
    return colored_heatmap

def using_device():
    """Set and print the device used for training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    return device

def setup_paths(data):
    """Set up data paths for training and validation."""
    folder_mapping = {
        "isic_2018_1": "isic_1/",
        "kvasir_1": "kvasir_1/",
        "ham_1": "ham_1/",
        "PH2Dataset": "PH2Dataset/",
        "isic_2016_1": "isic_2016_1/"
    }
    folder = folder_mapping.get(data)
    base_path = os.environ["ML_DATA_OUTPUT"] if torch.cuda.is_available() else os.environ["ML_DATA_OUTPUT_LOCAL"]
    print(base_path)
    return os.path.join(base_path, folder)

@torch.no_grad()
def update_teacher(student, teacher, momentum):
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data = momentum * param_t.data + (1. - momentum) * param_s.data
        

# Main Function
def main():
    # Configuration and Initial Setup
    
    data, training_mode, op = 'isic_2018_1', "ssl", "train"

    best_similarity = 0
    device      = using_device()
    folder_path = setup_paths(data)
    args, res   = parser_init("segmentation task", op, training_mode)
    res         = " ".join(res)
    res         = "["+res+"]"

    config      = wandb_init(os.environ["WANDB_API_KEY"], os.environ["WANDB_DIR"], args, data)

    # Data Loaders
    def create_loader(operation):
        return loader(operation,args.mode, args.sslmode_modelname, args.bsize, args.workers,
                      args.imsize, args.cutoutpr, args.cutoutbox, args.shuffle, args.sratio, data)

    train_loader    = create_loader(args.op)
    args.op         =  "validation"
    val_loader      = create_loader(args.op)
    args.op         = "train"

    # Student & Teacher modeli
    model   = model_dice_bce().to(device)
    student = model.encoder
    teacher = copy.deepcopy(student)
    teacher = teacher.to(device)
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    # Optimizasyon & Loss
    loss_fn         = DINOLoss()
    checkpoint_path = folder_path+str(model.__class__.__name__)+str(res)
    optimizer       = AdamW(student.parameters(), lr=config['learningrate'],weight_decay=0.05)
    scheduler       = CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learningrate'] / 10)
    
    print(f"Training on {len(train_loader) * args.bsize} images. Saving checkpoints to {folder_path}")
    print('Train loader global transform',train_loader.dataset.tr.global_transform,'Train loader local transform',train_loader.dataset.tr.local_transform)
    print(f"model config : {checkpoint_path}")

    # ML_DATA_OUTPUT      = os.environ["ML_DATA_OUTPUT"]+'isic_1/'
    # checkpoint_path_read = ML_DATA_OUTPUT+str(model.__class__.__name__)+str(res)
    # student.load_state_dict(torch.load(checkpoint_path_read, map_location=torch.device('cpu')))

    student_head = ProjectionHead().to(device)
    teacher_head = copy.deepcopy(student_head).to(device)
    for p in teacher_head.parameters():
        p.requires_grad = False
    # Training and Validation Loops
    def run_epoch(loader,epoch_idx, momentum,training=True):
        """Run a single training or validation epoch."""
        epoch_loss  = 0.0
        num_batches = 0
        epoch_val_loss = 0

        if not training:
            student.eval()
            teacher.eval()
        else:
            student.train()
        
        teacher_temp = get_teacher_temp(epoch_idx)
        current_lr = optimizer.param_groups[0]['lr']
        print("teacher temp",teacher_temp,"\n")
        print("current_lr",current_lr,"\n")

        with torch.set_grad_enabled(training):
            for student_augs, teacher_augs in loader:  #tqdm(loader, desc="Training" if training else "Validating", leave=False):

                loss        = 0.0

                student_feats = [student(im.to(device)) for im in student_augs]  # [B, C,H,W]
                student_pool = [im.mean(dim=(2,3)) for im in student_feats]     # [B, C]
                student_proj = [F.normalize(student_head(f), dim=1) for f in student_pool]

                with torch.no_grad():
                    teacher_feats = [teacher(im.to(device)) for im in teacher_augs] # [B, C,H,W]
                    teacher_pool = [im.mean(dim=(2,3)) for im in teacher_feats]     # [B, C]
                    teacher_proj = [F.normalize(teacher_head(f), dim=1) for f in teacher_pool]

                loss = loss_fn(student_proj, teacher_proj,teacher_temp)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    # Inside training loop, after loss.backward()
                    clip_grad_norm_(student.parameters(), max_norm=2.0)
                    optimizer.step()

                update_teacher(student, teacher, momentum)
                update_teacher(student_head, teacher_head, momentum)
                epoch_loss += loss.item()

                # Calculate cosine similarity during validation
                if not training:
    
                    # 2) Stack into [B, views, C]
                    stu_stack = torch.stack(student_pool, dim=1)   # [B, 8, C]
                    tea_stack = torch.stack(teacher_pool, dim=1)   # [B, 2, C]
                    # 3b) If you want all student vs all teacher pairwise:

                    pairwise = []
                    for s in range(6):
                        for t in range(2):
                            cos = torch.nn.functional.cosine_similarity(stu_stack[:, s], tea_stack[:, t], dim=-1)
                            pairwise.append(cos)
                    # `pairwise` is a list of 16 tensors [B]; stack and mean:
                    val_loss = torch.stack(pairwise, dim=1).mean()
                    epoch_val_loss += val_loss.item()
                    
                    # Select one sample (e.g. sample 1 from batch 1)
                    student_feat = student_feats[1][1]  # shape: [512, 8, 8]
                    teacher_feat = teacher_feats[1][1]  # shape: [512, 16, 16]

                    # Average across channels to get a single 8x8 or 16x16 map
                    student_map = student_feat.mean(dim=0)  # [8, 8]
                    teacher_map = teacher_feat.mean(dim=0)  # [16, 16]

                    # Convert to heatmap image
                    student_img = wandb.Image(featuremap_to_heatmap(student_map))
                    teacher_img = wandb.Image(featuremap_to_heatmap(teacher_map))

                    num_batches+=1
                    # Optionally log sample images to wandb
                    if num_batches == 1:  # just log the first batch to reduce clutter
                        # Log it
                        wandb.log({
                            "Val Sample - Student Aug": wandb.Image(student_augs[1][1]),
                            "Val Sample - Teacher Aug": wandb.Image(teacher_augs[1][1]),
                            "Val Sample - Student Output Heatmap": student_img,
                            "Val Sample - Teacher Output Heatmap": teacher_img,                          
                            })     

        if not training:
            return epoch_val_loss / len(loader)

        return epoch_loss / len(loader)

    epoch_idx=0
    for epoch in trange(config['epochs'], desc="Epochs"):

        # Training
        current_momentum = get_teacher_momentum(epoch, config['epochs'])
        train_loss = run_epoch(train_loader, epoch_idx, current_momentum,training=True )
        wandb.log({"Train Loss": train_loss})
        scheduler.step()

        cos_sim = run_epoch(val_loader, epoch_idx,current_momentum,training=False)
        wandb.log({"Cosine Similarity": cos_sim })

        epoch_idx+=1

        print("epoch_idx",epoch_idx,"\n")
        
        # Print losses and validation metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Cosine Similarity: {cos_sim:.4f}")

        # Save best model
        if cos_sim > best_similarity:
            best_similarity = cos_sim
            torch.save(student.state_dict(), checkpoint_path)
            print(f"Best model saved")

    wandb.finish()

if __name__ == "__main__":
    main()
