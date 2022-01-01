from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, time, copy, timm
from tqdm import tqdm
from typing import Optional, Union, ClassVar, Iterable, Any
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from PIL import Image
import segmentation_models_pytorch as smp
from torchvision.models import resnext50_32x4d
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import albumentations as A
import segmentation_models_pytorch as smp
from torchvision import transforms as T




class BrainSegmentorData(torch.utils.data.Dataset):
    def __init__(self, path: str, transform: torchvision.transforms) -> None:
        self.path = path
        self.transform = transform
        self.patients: list[str] = [
            folder for folder in os.listdir(self.path) if folder not in ['data.csv', 'README.md']
        ]
        self.csv_data: pd.DataFrame = pd.read_csv(os.path.join(self.path, 'data.csv'))
        self.masks: list[str] = []
        self.images: list[str] = []
        
        for patient in self.patients:
            for folder in os.listdir(os.path.join(self.path, patient)):
                if 'mask' in folder.split('.')[0].split('_'):
                    self.masks.append(os.path.join(self.path, patient, folder))
                else:
                    self.images.append(os.path.join(self.path, patient, folder))
        
        self.images.sort()
        self.masks.sort()
        self.brain_dict: dict[str, list[str]] = {
            'Patient': self.images, 
            'Images': self.images, 
            'Mask': self.masks
        }
        self.brain_dataframe: pd.DataFrame = pd.DataFrame.from_dict(self.brain_dict)
        self.brain_dataframe['diagnosis'] = self.brain_dataframe['Mask'].apply(
            lambda mask_path: self._diagnosis(mask_path)
        )
        
    
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    
    
    def __str__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Path', 'Transform'], [self.path, self.transform])
        })
    
        
    
    def __len__(self) -> int:
        return len(self.images)
    
    
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.images[index], self.masks[index]
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.transform:
            augumented = self.transform(image= image, mask= mask)
            image, mask = augumented['image'], augumented['mask']
        
        image = T.functional.to_tensor(image)
        mask = mask // 255
        mask = torch.Tensor(mask)
        return image, mask
    
    
    
    def _diagnosis(self, mask_path: str) -> int:
        value: int = np.max(Image.open(mask_path))
        return 1 if value > 0 else 0 



    def get_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.csv_data, self.brain_dataframe
    