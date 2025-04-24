import os
import torch
import wandb
import copy
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.data_loader import loader

from augmentation.Augmentation import Cutout, cutmix
from wandb_init import parser_init, wandb_init
from utils.metrics import calculate_metrics
from models.Model import model_dice_bce
from utils.Loss import DINOLoss
# from data.data_loader import DinoDataTransform

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
    return os.path.join(base_path, folder)


@torch.no_grad()
def update_teacher(student, teacher, momentum=0.996):
    for param_s, param_t in zip(student.parameters(), teacher.parameters()):
        param_t.data = momentum * param_t.data + (1. - momentum) * param_s.data
        

# Main Function
def main():
    # Configuration and Initial Setup
    
    data, training_mode, train = 'isic_2018_1', "ssl", True

    best_valid_loss = float("inf")
    device      = using_device()
    folder_path = setup_paths(data)
    args, res   = parser_init("segmentation task", "training", training_mode, train)
    res = " ".join(res)
    res = "["+res+"]"

    config      = wandb_init(os.environ["WANDB_API_KEY"], os.environ["WANDB_DIR"], args, data)
    
    
    # Data Loaders
    def create_loader(aug,train):
        return loader(args.mode, args.sslmode_modelname, train, args.bsize, args.workers,
                      args.imsize, args.cutoutpr, args.cutoutbox, aug, args.shuffle, args.sratio, data)

    train_loader = create_loader(args.aug,args.train)
    
    args.aug=False
    args.train=False
    val_loader   = create_loader(args.aug,args.train)
    args.aug=True
    args.train=True

    # Student & Teacher modeli
    model   = model_dice_bce().to(device)

    student = model.encoder
    teacher = copy.deepcopy(student)
    teacher.load_state_dict(student.state_dict())

    for p in teacher.parameters():
        p.requires_grad = False

    # Optimizasyon & Loss
    optimizer       = torch.optim.Adam(student.parameters(), lr=1e-4)
    loss_fn         = DINOLoss()
    checkpoint_path = folder_path+str(model.__class__.__name__)+str(res)
    optimizer       = Adam(model.parameters(), lr=config['learningrate'])
    scheduler       = CosineAnnealingLR(optimizer, config['epochs'], eta_min=config['learningrate'] / 10)

    print(f"Training on {len(train_loader) * args.bsize} images. Saving checkpoints to {folder_path}")
    print('Train loader transform',train_loader.dataset.tr)
    print('Val loader transform',val_loader.dataset.tr)
    print(f"model config : {checkpoint_path}")

    # Training and Validation Loops
    def run_epoch(loader, training=True):
        """Run a single training or validation epoch."""
        epoch_loss= 0.0
        num_batches = 0
        epoch_val_loss = 0
        student.train()
        with torch.set_grad_enabled(training):
            for student_augs, teacher_augs in tqdm(loader, desc="Training" if training else "Validating", leave=False):
                
                loss,n_pairs        = 0,0

                # 1. Get student outputs for all crops

                student_out = [ student(im) for im in student_augs ]  # list of length 8

                # 2. Get teacher outputs for just the first 2 global crops
                with torch.no_grad():
                    teacher_out = [ teacher(teacher_augs[0]), teacher(teacher_augs[1]) ]  # list of length 2


                for s in student_out:
                    for t in teacher_out:
                        loss += loss_fn(s, t)
                        n_pairs    += 1

                # 4) Final average:
                loss = loss / n_pairs   # averages over the 6×2 = 12 pairs
                
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                update_teacher(student, teacher)
                epoch_loss += loss.item()

                # Calculate cosine similarity during validation
                if not training:
    
                    # 1) Pool spatial dims
                    stu_pooled = [s.mean(dim=(2,3)) for s in student_out]   # each [B, C]
                    tea_pooled = [t.mean(dim=(2,3)) for t in teacher_out]   # each [B, C]

                    # 2) Stack into [B, views, C]
                    stu_stack = torch.stack(stu_pooled, dim=1)   # [B, 8, C]
                    tea_stack = torch.stack(tea_pooled, dim=1)   # [B, 2, C]
                    # 3b) If you want all student vs all teacher pairwise:

                    pairwise = []
                    for s in range(6):
                        for t in range(2):
                            cos = torch.nn.functional.cosine_similarity(stu_stack[:, s], tea_stack[:, t], dim=-1)
                            pairwise.append(cos)
                    # `pairwise` is a list of 16 tensors [B]; stack and mean:
                    val_loss = torch.stack(pairwise, dim=1).mean()
                    epoch_val_loss += val_loss.item()

                    num_batches+=1
                    # Optionally log sample images to wandb
                    if num_batches == 1:  # just log the first batch to reduce clutter
                        wandb.log({
                            "Val Sample - Input x": [wandb.Image(student_augs[i]) for i in range(min(2, student_augs.shape[0]))],
                            "Val Sample - Input y": [wandb.Image(teacher_augs[i]) for i in range(min(2, teacher_augs.shape[0]))],
                            "Val Sample - Student Output ": wandb.Image(student_out[1][1]),
                            "Val Sample - Teacher Output ": wandb.Image(teacher_out[1][1])
                        })

        if not training:
            return epoch_val_loss / len(loader)

        return epoch_loss / len(loader)

    for epoch in trange(config['epochs'], desc="Epochs"):

        # Training
        train_loss = run_epoch(train_loader, training=True)
        wandb.log({"Train Loss": train_loss})

        val_metric = run_epoch(val_loader, training=False)
        wandb.log({"Cosine Similarity": val_metric })

        # Print losses and validation metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Cosine Similarity: {val_metric:.4f}")

        # Save best model
        if val_metric < best_valid_loss:
            best_valid_loss = val_metric
            torch.save(student.state_dict(), checkpoint_path)
            print(f"Best model saved with cosine sim: {val_metric:.4f}")

    wandb.finish()

if __name__ == "__main__":
    main()
