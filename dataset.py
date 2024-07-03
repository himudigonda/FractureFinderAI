import os
import pandas as pd
import cv2
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Classification Dataset
class ChestXRayClassificationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.annotations.iloc[idx, 1]
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

# Localization Dataset
class ChestXRayLocalizationDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = self.annotations.iloc[idx, 1:].values.reshape(-1, 4)
        
        if self.transform:
            augmented = self.transform(image=image, bboxes=boxes)
            image = augmented['image']
            boxes = augmented['bboxes']
        
        return image, boxes

# Data Augmentations
def get_train_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

def get_valid_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=[]))

# Data Loaders
def get_classification_dataloader(csv_file, root_dir, batch_size=32, train=True):
    if train:
        transforms = get_train_transforms()
    else:
        transforms = get_valid_transforms()
    
    dataset = ChestXRayClassificationDataset(csv_file, root_dir, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

def get_localization_dataloader(csv_file, root_dir, batch_size=32, train=True):
    if train:
        transforms = get_train_transforms()
    else:
        transforms = get_valid_transforms()
    
    dataset = ChestXRayLocalizationDataset(csv_file, root_dir, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)
