from torch.utils.data import DataLoader

from data.Custom_Dataset import dataset
from glob import glob
from torchvision.transforms import v2 
import os
import torch

def data_transform():
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    # ----- GLOBAL CROPS (2 views) -----
    global_transforms = v2.Compose([
        # larger crop, encourages invariance to scale
        v2.RandomResizedCrop(256, scale=(0.6, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),

        # color distortions
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),

        # blur + solarize
        v2.RandomApply([v2.GaussianBlur(
            kernel_size=int(0.1 * 256)//2*2 + 1, sigma=(0.1, 2.0)
        )], p=1.0),
        v2.RandomSolarize(threshold=0.5, p=0.2),

        # final normalization
        v2.ToTensor(),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # ----- LOCAL CROPS (6 views) -----
    local_transforms = v2.Compose([
        # smaller crop, focuses on fine details
        v2.RandomResizedCrop(128, scale=(0.1, 0.6), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),

        # same color jitter but maybe slightly weaker
        v2.RandomApply([v2.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        v2.RandomGrayscale(p=0.2),

        # occasional blur
        v2.RandomApply([v2.GaussianBlur(
            kernel_size=int(0.05 * 256)//2*2 + 1, sigma=(0.1, 2.0)
        )], p=0.5),

        v2.ToTensor(),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Builds 2 global + 6 local crops per image:
    return DinoMultiCropTransform(global_transforms,
                                  local_transforms,
                                  n_global_crops=2,
                                  n_local_crops=6)

class DinoMultiCropTransform:
    def __init__(self, global_transform, local_transform, n_local_crops=6,n_global_crops=2):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.n_local = n_local_crops
        self.n_global_crops = n_global_crops

    def __call__(self, img):
        student_crops = []
        teacher_crops = []
        # 2 global views
        for _ in range(self.n_global_crops):
            teacher_crops.append(self.global_transform(img))
        # n local views
        for _ in range(self.n_local):
            student_crops.append(self.local_transform(img))
        
        return student_crops,teacher_crops

def loader(op,mode,sslmode,batch_size,num_workers,image_size,cutout_pr,cutout_box,shuffle,split_ratio,data):

    if data=='isic_2018_1':
        foldernamepath="isic_2018_2/"
        imageext="/*.jpg"
        maskext="/*.png"
    elif data == 'kvasir_1':
        foldernamepath="kvasir_1/"
        imageext="/*.jpg"
        maskext="/*.jpg"
    elif data == 'ham_1':
        foldernamepath="HAM10000_1/"
        imageext="/*.jpg"
        maskext="/*.png"
    elif data == 'PH2Dataset':
        foldernamepath="PH2Dataset/"
        imageext="/*.jpeg"
        maskext="/*.jpeg"
    elif data == 'isic_2016_1':
        foldernamepath="isic_2016_1/"
        imageext="/*.jpg"
        maskext="/*.png"

    if op =="train":
        train_im_path   = os.environ["ML_DATA_ROOT"]+foldernamepath+"train/images"   
        train_mask_path = os.environ["ML_DATA_ROOT"]+foldernamepath+"train/masks"
        
        train_im_path   = sorted(glob(train_im_path+imageext))
        train_mask_path = sorted(glob(train_mask_path+maskext))
    
    elif op == "validation":
        test_im_path    = os.environ["ML_DATA_ROOT"]+foldernamepath+"val/images"
        test_mask_path  = os.environ["ML_DATA_ROOT"]+foldernamepath+"val/masks"
        test_im_path    = sorted(glob(test_im_path+imageext))
        test_mask_path  = sorted(glob(test_mask_path+maskext))

    else :
        test_im_path    = os.environ["ML_DATA_ROOT"]+foldernamepath+"test/images"
        test_mask_path  = os.environ["ML_DATA_ROOT"]+foldernamepath+"test/masks"
        test_im_path    = sorted(glob(test_im_path+imageext))
        test_mask_path  = sorted(glob(test_mask_path+maskext))

    transformations = data_transform()

    if torch.cuda.is_available():
        if op == "train":
            data_train  = dataset(train_im_path,train_mask_path,cutout_pr,cutout_box, transformations,mode)
        else:
            data_test   = dataset(test_im_path, test_mask_path,cutout_pr,cutout_box, transformations,mode)

    elif op == "train":  #train for debug in local
        data_train  = dataset(train_im_path[10:20],train_mask_path[10:20],cutout_pr,cutout_box, transformations,mode)

    else:  #test in local
        data_test   = dataset(test_im_path[10:20], test_mask_path[10:20],cutout_pr,cutout_box, transformations,mode)

    if op == "train":
        train_loader = DataLoader(
            dataset     = data_train,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
            persistent_workers=True
            )
        return train_loader
    
    else :
        test_loader = DataLoader(
            dataset     = data_test,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
        )
    
        return test_loader


#loader()