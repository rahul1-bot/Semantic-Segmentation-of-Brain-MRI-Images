from __future__ import annotations

__Patent_license__: str = r'''
    MIT License
    Copyright (c) 2021 Rahul Sawhney
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''

__authors__: list[str] = ['Rahul Sawhney', 'Shubham Garg', 'Harsh Sharma', 'Aabha Malik']

__authors_qualifications__: str = r'''
    Rahul Sawhney: Btech CSE, Amity University, Noida 
    Shubham Garg: Btech CSE,  Amity University, Noida
    Harsh Sharma: Btech CSE,  Amity University, Noida
    Aabha Malik: Btech CSE,   Amity University, Noida
'''

__Patent_doc__: str = r'''
    >>> Patent Topic: Semantic Segmentation of Brain MRI Images for FLAIR Abnormailty Detection  
    >>> Patent Abstract: //...//
    >>> Patent Conclusion: //...//
'''

__loss_doc__: str = r'''
    LOSS = BCE - Log(DICE)
    where,
        1) BCE = Binary Cross Entropy Loss Function
        2) DICE = Custom Dice Coeffient Loss 
'''


import warnings, os, copy, time
from tqdm import tqdm
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils



class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, transform: torchvision.transforms) -> None:
        self.path = path
        self.transform = transform
        self.dataset: pd.DataFrame = self.get_data()


    
    def get_data(self) -> pd.DataFrame:
        patients_list: list[str] = [
            folder for folder in os.listdir(self.path) if folder not in ['data.csv', 'README.md']
        ]
        images: list[str] = []
        masks: list[str] = []
        data_map: dict[str, list[str]] = {
            x: [] for x in ['patient', 'image_path', 'mask_path']
        }
        for patient in patients_list:
            for patient_file in os.listdir(os.path.join(self.path, patient)):
                if 'mask' in patient_file.split('.')[0].split('_'):
                    data_map['mask_path'].append(os.path.join(self.path, patient, patient_file))
                    data_map['patient'].append(patient)
                else:
                    data_map['image_path'].append(os.path.join(self.path, patient, patient_file))

        data_map: pd.DataFrame = pd.DataFrame(data_map)        
        data_map['diagnosis'] = data_map['mask_path'].apply(lambda mask_path: BrainDataset._apply_diagnosis(mask_path))
        return data_map



    def __len__(self) -> int:
        return len(self.dataset)
    


    def __repr__(self) -> str:
        return str({
            x: y for x, y in zip(['Module', 'Name', 'Object_ID'], [self.__module__, type(self).__name__, hex(id(self))])
        })


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image: str = self.dataset['image_path'][index]
        mask: str = self.dataset['mask_path'][index]
        if self.transform:
            image, mask = Image.open(image), Image.open(mask)
            image, mask = self.transform(image), self.transform(mask)
        
        return image, mask



    @staticmethod
    def _apply_diagnosis(mask_path: str) -> int:
        mask = Image.open(mask_path)
        val: int = np.max(mask)
        return 1 if val > 0 else 0
        
    


class BrainAnalysis:
    data_map: ClassVar[dict[int, str]] = {
        0: 'No Tumor',
        1: 'Tumor'
    }

    def __init__(self, dataset: pd.DataFrame) -> None:
        self.dataset = dataset


    def __repr__(self) -> str:
        return str({
            x: y for x, y in zip(['Module', 'Name', 'Object_ID'], [self.__module__, type(self).__name__, hex(id(self))])
        })


    def data_distribution(self) -> plot():
        axes = self.dataset['diagnosis'].value_counts().plot(
            kind= 'bar', 
            stacked= True, 
            figsize= (10, 6), 
            color= ['violet', 'lightseagreen']
        )
        axes.set_xticklabels(self.data_map.values(), rotation= 45, fontsize= 9)
        axes.set_ylabel('Total Images', fontsize= 9)
        axes.set_title('Distribution of dataset')
        plt.show()
    



class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int) -> None:
        super(ConvBlock, self).__init__()
        layers: dict[str, object] = {
            'conv_1': nn.Conv2d(in_channels, out_channels, kernel_size, padding= padding, bias= False),
            'batch_norm': nn.BatchNorm2d(out_channels, eps= 1e-4),
            'relu': nn.ReLU(inplace = True)
        }
        self.block = nn.Sequential(*layers.values())
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)




class StackEncoder(nn.Module):
    def __init__(self, channel_1: int, channel_2: int, kernel_size: Optional[int] = 3, 
                                                       padding: Optional[int] = 1) -> None:
        super(StackEncoder, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.block = nn.Sequential(
            ConvBlock(channel_1, channel_2, kernel_size, padding),
            ConvBlock(channel_2, channel_2, kernel_size, padding)
        )


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.block(x)
        pool_x = self.pool(x)
        return x, pool_x

        


class StackDecoder(nn.Module):
    def __init__(self, big_channel: int, channel_1: int, channel_2: int, 
                                                         kernel_size: Optional[int] = 3, 
                                                         padding: Optional[int] = 1) -> None:
        super(StackDecoder, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channel_1 + big_channel, channel_2, kernel_size, padding),
            ConvBlock(channel_2, channel_2, kernel_size, padding),
            ConvBlock(channel_2, channel_2, kernel_size, padding)
        )
    

    def forward(self, x: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = down.size()
        x = F.upsample(x, size= (height, width), mode= 'bilinear')
        x = torch.cat([x, down], 1)
        x = self.block(x)
        return x




class UNet(nn.Module):
    def __init__(self, in_channels: tuple[int, ...]) -> None:
        super(UNet, self).__init__()
        channel, height, width = in_channels 
        self.down_one = StackEncoder(channel, 12)
        self.down_two = StackEncoder(12, 24)
        self.down_three = StackEncoder(24, 46)
        self.down_four = StackEncoder(46, 64)
        self.down_five = StackEncoder(64, 128)

        self.centre = ConvBlock(128, 128, kernel_size= 3, padding= 1)

        self.up_five  = StackDecoder(128, 128, 64)  
        self.up_four  = StackDecoder(64, 64, 46)
        self.up_three = StackDecoder(46, 46, 24)  
        self.up_two   = StackDecoder(24, 24, 12)    
        self.up_one   = StackDecoder(12, 12, 12)

        self.conv = nn.Conv2d(12, 1, kernel_size= 1, bias= True)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_one, out = self.down_one(x)
        down_two, out = self.down_two(out)
        down_three, out = self.down_three(out)
        down_four, out = self.down_four(out)
        down_five, out = self.down_five(out)

        out = self.centre(out)

        up_five = self.up_five(out, down_five)
        up_four = self.up_four(up_five, down_four)
        up_three = self.up_three(up_four, down_three)
        up_two = self.up_two(up_three, down_two)
        up_one = self.up_one(up_two, down_one)

        out = self.conv(up_one)
        return out




class Metric(nn.Module):
    def __init__(self) -> None:
        super(Metric, self).__init__()
    

    def forward(self, inputs: np.ndarray, target: np.ndarray) -> float:
        intersection = 2.0 * (target * inputs).sum()
        union = target.sum() + inputs.sum()
        if target.sum() == 0 and inputs.sum() == 0:
            return 1.0
        return intersection / union




class Loss(nn.Module):
    def __init__(self) -> None:
        super(Loss, self).__init__()
    

    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return Loss._bce_dice_loss(inputs, target)
    

    @staticmethod
    def _dice_coeff_loss(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth: float = 1.0
        intersection = 2.0 * ((target * inputs).sum()) + smooth
        union = target.sum() + inputs.sum() + smooth
        return 1 - (intersection / union)
    
    
    @staticmethod
    def _bce_dice_loss(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_score = Loss._dice_coeff_loss(inputs, target)
        bce_score = nn.BCELoss()
        bce_loss = bce_score(inputs, target)
        return bce_loss + dice_score




class Model():
    def __init__(self, model_name: str, model: object, 
                                        train_loader: object, 
                                        test_loader: object,
                                        criterion: object,
                                        optimizer: object,
                                        num_epochs: int,
                                        metric: object,
                                        device: torch.device, 
                                        lr_scheduler: Optional[bool] = False) -> None:
        super(Model, self).__init__()
        self.model_name = model_name
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.metric = metric
        self.device = device
        
    
    def train_validate(self) -> 


if __name__.__contains__('__main__'):
    path: str = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\brain_segmentation\\kaggle_3m'
    
    transforms_list: list = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomRotation(360),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ])

    brain_data = BrainDataset(path, transforms_list)
    train_data, test_data = torch.utils.data.random_split(brain_data, [3600, 329])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size= 4, shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= 4)

    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_unet = UNet((3, 256, 256)).to(device)
    criterion = Loss()
    metric = Metric()
    optimizer = torch.optim.Adam(model_unet.parameters(), lr= 1e-3)

    image_segmentor = Model(
        model_name= 'unet', model= model_unet, train_loader=  train_loader, test_loader= test_loader, criterion= criterion, optimizer= optimizer, num_epochs= 2, metric= metric, device= device
    )

    image_segmentor.train_validate()





    #BrainAnalysis(brain_data.dataset).data_distribution()
    