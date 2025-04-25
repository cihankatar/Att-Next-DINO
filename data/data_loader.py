from torch.utils.data import DataLoader

from data.Custom_Dataset import dataset
from utils.Test_Train_Split import ssl_data_split
from glob import glob
from torchvision.transforms import v2 
import os
import torch



def data_transform():
        
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)

    global_transforms = v2.Compose([
        v2.RandomResizedCrop(256, scale=(0.4, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
        v2.RandomGrayscale(p=0.2),
        v2.GaussianBlur(kernel_size= int(0.1*256) // 2 * 2 + 1, sigma=(0.1, 2.0)),
        v2.ToTensor(),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),])

    local_transforms = v2.Compose([
        v2.RandomResizedCrop(256//2, scale=(0.05, 0.4), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
        v2.RandomGrayscale(p=0.2),
        v2.GaussianBlur(kernel_size= int(0.05*256) // 2 * 2 + 1, sigma=(0.1, 2.0)),
        v2.ToTensor(),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    
    transformations = DinoMultiCropTransform(global_transforms, local_transforms, n_local_crops=6)

    return transformations


class DinoMultiCropTransform:
    def __init__(self, global_transform, local_transform, n_local_crops=6):
        self.global_transform = global_transform
        self.local_transform = local_transform
        self.n_local = n_local_crops

    def __call__(self, img):
        student_crops = []
        teacher_crops = []
        # 2 global views
        for _ in range(2):
            teacher_crops.append(self.global_transform(img))
        # n local views
        for _ in range(self.n_local):
            student_crops.append(self.local_transform(img))
        
        return student_crops,teacher_crops

"""def data_transform(mode,task,train,image_size):

    transformations = v2.Compose([
        v2.RandomResizedCrop([image_size,image_size],antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(0.4, 0.4, 0.4, 0.1),
        v2.RandomGrayscale(p=0.2),
        #v2.ToTensor(),
        #v2.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
            
    transformations = DinoDataTransform(transformations)

    return transformations

    

class DinoDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
 
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


"""

def loader(mode,sslmode,train,batch_size,num_workers,image_size,cutout_pr,cutout_box,aug,shuffle,split_ratio,data):
    
    if data=='isic_2018_1':
        foldernamepath="isic_2018_1/"
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

    train_im_path   = os.environ["ML_DATA_ROOT"]+foldernamepath+"train/images"   
    train_mask_path = os.environ["ML_DATA_ROOT"]+foldernamepath+"train/masks"
    
    if train:
        test_im_path    = os.environ["ML_DATA_ROOT"]+foldernamepath+"val/images"
        test_mask_path  = os.environ["ML_DATA_ROOT"]+foldernamepath+"val/masks"
    else :
        test_im_path    = os.environ["ML_DATA_ROOT"]+foldernamepath+"test/images"
        test_mask_path  = os.environ["ML_DATA_ROOT"]+foldernamepath+"test/masks"


    train_im_path   = sorted(glob(train_im_path+imageext))
    train_mask_path = sorted(glob(train_mask_path+maskext))
    test_im_path    = sorted(glob(test_im_path+imageext))
    test_mask_path  = sorted(glob(test_mask_path+maskext))
    print(train_im_path)

    transformations = data_transform()

    if torch.cuda.is_available():
        if train:
            data_train  = dataset(train_im_path,train_mask_path,cutout_pr,cutout_box, aug, transformations,mode)
        else:
            data_test   = dataset(test_im_path, test_mask_path,cutout_pr,cutout_box, aug, transformations,mode)

    elif train:  #train for debug in local
        data_train  = dataset(train_im_path[10:20],train_mask_path[10:20],cutout_pr,cutout_box, aug, transformations,mode)

    else:
        data_test   = dataset(test_im_path[10:20], test_mask_path[10:20],cutout_pr,cutout_box, aug, transformations,mode)

    if train:
        train_loader = DataLoader(
            dataset     = data_train,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = num_workers,
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