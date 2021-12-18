from __future__ import annotations

__authors__: list[str] = ['Rahul Sawhney', 'Aabha Malik', 'Shubham Garg', 'Harsh Sharma']

__authors_email__: dict[str, str] = {
    'Rahul Sawhney': 'sawhney.rahulofficial@outlook.com',
    'Aabha Malik': 'aabhamalik30@gmail.com',
    'Shubham Garg': '',
    'Harsh Sharma': ''
}

__authors_qualifications__: dict[str, str] = {
    x: 'Btech CSE, Amity University, Noida' 
    for x in ['Rahul Sawhney', 'Aabha Malik', 'Shubham Garg', 'Harsh Sharma']    

}


__license__: str = r'''
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

__doc__: str = r'''
    >>> Paper Title: 
            **Semantic Segmentation of brain Images from UNet, feature Pyramid Network 
    
    >>> Paper Abstract: 
            Brain Tumor segmentation is an essential method in medical image processing. 
            Early medical diagnosis of brain tumors plays an essential function in enhancing 
            treatment possibilities and increases the survival rate of the patients. The 
            most challenging and time taking work is the manual segmentation of the brain 
            tumors for cancer diagnosis from large quantity of MRI images produced in 
            scientific routine. Recently, automated segmentation utilizing deep learning 
            techniques showed popular considering that these techniques accomplish cutting
            edge outcomes and resolve this issue much better than other approaches. 
            Deep Learning techniques can likewise make it possible for effective processing
            and unbiased assessment of the big quantities of MRI-based image information. 
            There are variety of existing evaluation papers, concentrating on conventional 
            techniques for MRI-based brain tumor image segmentation. In this paper, we have 
            used UNet model along with this the **AdamW and **AdaMax are used as optimizer. 
            Criterion used is IOU (Intersection Over Union) and the metric utilized for evaluation
            is Dice Score. 
    
    >>> Paper Keywords: 
        Lower-grade gliomas detection, Brain Semantic Segmentation, Image Recogonition, UNet, Feature-Detector-Network
'''



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os, time, tqdm, copy
from typing import Optional, Union, ClassVar
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils
from PIL import Image
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau




class BrainSegmentorData(torch.utils.data.Dataset):
    def __init__(self, path: str, transform: torchvision.transforms) -> None:
        self.path = path
        self.transform = transform
        self.patients: list[str] = [
            folder for folder in os.listdir(self.path) if folder not in ['data.csv', 'README.md']
        ]
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
        self.brain_dict: dict[str, list[str]] = {'Patient': self.images, 'Images': self.images, 'Mask': self.masks}
        self.brain_dataframe: pd.DataFrame = pd.DataFrame.from_dict(self.brain_dict)
        self.brain_dataframe['diagnosis'] = self.brain_dataframe['Mask'].apply(lambda mask_path: self._diagnosis(mask_path))
            
    
    
    def __repr__(self) -> str(dict[str, str]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    
    
    def __str__(self) -> str(dict[str, str]):
        return str({
            x: y for x, y in zip(['Path', 'Transform'], [self.path, self.transform])
        })
    
        
    
    def __len__(self) -> int:
        return len(self.images)
    
    
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.images[index], self.masks[index]
        if self.transform is not None:
            image, mask = Image.open(image_path), Image.open(mask_path)
            image, mask = self.transform(image), self.transform(mask)
        return image, mask
    
    
    
    def _diagnosis(self, mask_path: str) -> int:
        value: int = np.max(Image.open(mask_path))
        return 1 if value > 0 else 0 





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

    
    def __str__(self) -> str(dict[str, int]):
        return str(self.data_map)
    

    def data_distribution(self) -> 'plot':
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
    def __init__(self, in_channels: int, out_channels: int, 
                                         kernel_size: Optional[int] = 3, 
                                         padding: Optional[int] = 1) -> None:
        super(ConvBlock, self).__init__()
        layers: dict[str, object] = {
            'conv': nn.Conv2d(in_channels, out_channels, kernel_size, padding= padding, bias= False),
            'batch_norm': nn.BatchNorm2d(out_channels, eps= 1e-4),
            'relu': nn.ReLU(inplace= True)
        }
        self.block = nn.Sequential(*layers.values())
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)




class StackEncoder(nn.Module):
    def __init__(self, channel_one: int, channel_two: int, kernel_size: Optional[int] = 3, padding: Optional[int] = 1) -> None:
        super(StackEncoder, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size= 2, stride= 2)
        layers: dict[str, object] = {
            'conv_one': ConvBlock(channel_one, channel_two, kernel_size, padding),
            'conv_two': ConvBlock(channel_two, channel_two, kernel_size, padding)
        }
        self.block = nn.Sequential(*layers.values())
    
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        big_out: torch.Tensor = self.block(x)
        pool_out: torch.Tensor = self.max_pool(big_out)
        return big_out, pool_out
        
        


class StackDecoder(nn.Module):
    def __init__(self, big_channel: int, channel_one: int, channel_two: int, 
                                                           kernel_size: Optional[int] = 3, 
                                                           padding: Optional[int] = 1) -> None:
        super(StackDecoder, self).__init__()
        layers: dict[str, object] = {
            'conv_one': ConvBlock(channel_one + big_channel, channel_two, kernel_size, padding),
            'conv_two': ConvBlock(channel_two, channel_two, kernel_size, padding),
            'conv_three': ConvBlock(channel_two, channel_two, kernel_size, padding)
        }
        self.block = nn.Sequential(*layers.values())
    
    
    
    def forward(self, x: torch.Tensor, down_tensor: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = down_tensor.size()
        x = F.upsample(x, size= (height, width), mode= 'bilinear')
        x = torch.cat([x, down_tensor], 1)
        x = self.block(x)
        return x 
    
       
       

class UNet(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int]) -> None:
        super(UNet, self).__init__()
        channel, height, width = input_shape
        
        self.down_one = StackEncoder(channel, 12)
        self.down_two = StackEncoder(12, 24)
        self.down_three = StackEncoder(24, 46)
        self.down_four = StackEncoder(46, 64)
        self.down_five = StackEncoder(64, 128)
        
        self.centre = ConvBlock(128, 128)
        
        self.up_five = StackDecoder(128, 128, 64)
        self.up_four = StackDecoder(64, 64, 46)
        self.up_three = StackDecoder(46, 46, 24)
        self.up_two = StackDecoder(24, 24, 12)
        self.up_one = StackDecoder(12, 12, 12)
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
                            



class Feature_Pyramid_Network(nn.Module):
    def __init__(self) -> None:
        super(Feature_Pyramid_Network, self).__init__()
        
    
    def forward(self) -> nn.Module():
        model = smp.FPN(
            encoder_name= 'efficientnet-b7',
            encoder_weights= 'imagenet',
            in_channels= 3,
            classes= 1,
            activation= 'sigmoid',
        )
        return model
    
    
        

class IOULoss(nn.Module):
    def __init__(self) -> None:
        super(IOULoss, self).__init__()
    
    
    def forward(self, predictions: torch.Tensor, masks: torch.Tensor, e: Optional[float] = 1e-7) -> float:
        predictions = torch.where(predictions > 0.5, 1, 0)
        masks = masks.byte()
        intersection = (predictions & masks).float().sum((1, 2))
        union = (predictions | masks).float().sum((1, 2))
        iou = (intersection + e) / (union + e)
        return iou




class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss, self).__init__()
    
    
    def forward(self, predictions: torch.Tensor, masks: torch.Tensor, e: Optional[float] = 1e-7) -> float:
        predictions = torch.where(predictions > 0.5, 1, 0)
        masks = masks.byte()
        intersection = (predictions & masks).float().sum((1, 2))
        return ((2 * intersection) + e) / (predictions.float().sum((1, 2)) + masks.float().sum((1, 2)) + e)
            
            
            

class BCEDice(nn.Module):
    def __init__(self) -> None:
        super(BCEDice, self).__init__()
        
    
    def forward(self, output: torch.Tensor, target: torch.Tensor, alpha: Optional[float] = 0.01) -> float:
        bce = torch.nn.functional.binary_cross_entropy(output, target)
        soft_dice = 1 - dice_pytorch(output, target).mean()
        return bce + alpha * soft_dice
        
        
        

class EarlyStop:
    def __init__(self, patience: Optional[int] = 6, min_delta: Optional[float] = 0, 
                                                    weights_path: Optional[str] = 'model_weights.pt') -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter: int = 0
        self.best_loss: float = float('inf')
        self.weights_path = weights_path
    
    
    def __repr__(self) -> str(dict[str, str]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    
    def __str__(self) -> str(dict[str, str]):
        return str({
            'Patience': self.patience,
            'Min_Delta': self.min_delta,
            'Counter': self.counter,
            'Optimal_loss': self.best_loss,
            'Model_weights': self.weights_path
        })
        
        
    
    def __call__(self, test_loss: float, model: torch.nn.Module) -> bool:
        if self.best_loss - test_loss > self.min_delta:
            self.best_loss = test_loss
            torch.save(model.state_dict(), self.weights_path)
            self.counter = 0
        
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        
        return False
    
    
    
    def load_weights(self, model: torch.nn.Module) -> torch.nn.Module():
        return model.load_state_dict(torch.load(self.weights_path))
    
    



class Model():
    def __init__(self, epochs: int, model: torch.nn.Module(), train_loader: object, 
                                                              test_loader: object, 
                                                              optimizer: object,
                                                              iou_loss: object,
                                                              dice_loss: object, 
                                                              criterion: object, 
                                                              lr_schedular: torch.optim.lr_scheduler,
                                                              device: torch.device) -> None:
        super(Model, self).__init__()
        self.epochs = epochs
        self.model = Model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.iou_loss = iou_loss
        self.dice_loss = dice_loss
        self.criterion = criterion
        self.lr_schedular = lr_schedular
        self.device = device
    


    def __repr__(self) -> str(dict[str, str]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
        
    
    def __str__(self) -> str(dict[str, str]):
        return str({
            'epochs': self.epochs,
            'model': self.model,
            'train_loader': self.train_loader,
            'test_loader': self.test_loader,
            'optimizer': self.optimizer,
            'iou_loss': self.iou_loss,
            'dice_loss': self.dice_loss,
            'criterion': self.criterion,
            'lr_schedular': self.lr_schedular,
            'device': self.device 
        })
        
    
    def train_validate(self, history: Optional[bool] = False) -> Union[None, dict[str, float]]:
        self.history: dict[str, list] = {
            x: [] for x in ['train_loss', 'test_loss', 'test_iou', 'test_dice']
        }
        early_stop: object = EarlyStop(patience= 7)
        
        for epoch in range(1, self.epochs + 1):
            start_time: float = time.time()
            running_loss: float = 0.0
            self.model.train()
            
            for index, data in enumerate(tqdm(self.train_loader)):
                img, mask = data
                img, mask = img.to(self.device), mask.to(self.device)
                predictions = self.model(img)
                predictions = predictions.squeeze(1)
                loss = self.criterion(predictions, mask)
                running_loss += loss.item() * img.size(0)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            self.model.eval()
            
            with torch.no_grad():
                running_IoU: float = 0.0
                running_dice: float = 0.0
                running_valid_loss: float = 0.0
                
                for index, data in enumerate(self.test_loader):
                    img, mask = data
                    img, mask = img.to(self.device), mask.to(self.device)
                    predictions = self.model(img)
                    predictions = predictions.squeeze(1)
                    running_dice += self.dice_loss(predictions, mask).sum().item()
                    running_IoU += self.iou_loss(predictions, mask).sum().item()
                    loss = self.criterion(predictions, mask)
                    running_valid_loss += loss.item() * img.size(0)
            
            train_loss = running_loss / len(self.train_loader.dataset)
            test_loss = running_valid_loss / len(self.test_loader.dataset)
            test_dice = running_dice / len(self.test_loader.dataset)
            test_iou = running_IoU / len(self.test_loader.dataset)
            
            history['train_loss'].append(train_loss)
            history['test_loss'].append(test_loss)
            history['test_iou'].append(test_iou)
            history['test_dice'].append(test_dice)
            
            print(f'Epoch: {epoch}/{epochs} | Training loss: {train_loss} | Validation loss: {val_loss} | Validation Mean IoU: {val_IoU} '
            f'| Validation Dice coefficient: {val_dice}')
            
            self.lr_scheduler.step(test_loss)
            if early_stop(test_loss, self.model):
                early_stop.load_weights(self.model)
                break
    
        self.model.eval()
    
        if self.histroy:
            return history
        


    def train_test_loss(self, fig_size: Optional[tuple[int, int]] = (7, 7)) -> 'plot':
        plt.figure(figsize= fig_size) 
        plt.plot(self.history['train_loss'], label= 'Train loss')
        plt.plot(self.history['test_loss'], label= 'Test loss')
        plt.ylim(0, 0.01)
        plt.legend()
        plt.show()
    
    
    
    def test_Iou_Dice(self, fig_size: Optional[tuple[int, int]] = (7, 7)) -> 'plot':
        plt.figure(figsize= fig_size)
        plt.plot(self.history['test_iou'], label= 'Testing Mean Jaccard index')
        plt.plot(history['test_dice'], label= 'Testing Dice coefficient')
        plt.legend()
        plt.show()
        
        
        
        
        
#@: Driver code
if __name__.__contains__('__main__'):
    path: str = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\brain_segmentation\\kaggle_3m'
    transforms_list = transforms.Compose([
        transforms.Resize(256),
        #transforms.RandomCrop(256),
        #transforms.RandomRotation(360),
        #transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.5), (0.5))
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    brain_data: object = BrainSegmentorData(path, transforms_list)
    #print(type(brain_data.show()))
    #print(len(brain_data))
    print(brain_data.brain_dataframe)
    
    for image, mask in brain_data:
        print(image.shape)
        print(mask.shape)
        break
    
    train_data, test_data = torch.utils.data.random_split(brain_data, [3600, 329])
    
    train_loader = torch.utils.data.DataLoader(dataset= train_data, batch_size= 4, shuffle= True)
    test_loader = torch.utils.data.DataLoader(dataset= test_data, batch_size= 4)
    
    device: torch.device = ('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    unet = UNet((3, 256, 256)).to(device)
    feature_network = Feature_Pyramid_Network().to(device)
    
    criterion = BCEDice
    optimizer_unet = torch.optim.Adam(unet.parameters(), lr= 0.001)
    optimizer_feature_network = torch.optim.adam(feature_network.parameters(), lr= 0.001)
    
    lr_schedular_unet = ReduceLROnPlateau(optimizer= optimizer_unet, patience= 2, factor= 0.2)
    lr_schedular_feature = ReduceLROnPlateau(optimizer= optimizer_feature_network, patience= 2, factor= 0.2)
    epochs: int = 5
    
    iou_loss = IOULoss
    dice_loss = DiceLoss
    
    
    model_unet = Model(
        epochs= epochs,
        model= unet,
        train_loader = train_loader,
        test_loader = test_loader,
        optimizer = optimizer_unet,
        iou_loss = iou_loss,
        dice_loss = dice_loss,
        criterion = criterion,
        lr_schedular = lr_schedular_unet,
        device = device
    )
    
    model_feature_network = Model(
        epochs= epochs,
        model= feature_network,
        train_loader = train_loader,
        test_loader = test_loader,
        optimizer = optimizer_feature_network,
        iou_loss = iou_loss,
        dice_loss = dice_loss,
        criterion = criterion,
        lr_schedular = lr_schedular_feature,
        device = device
    )
    
    model_unet.train_validate()
    model_feature_network.train_validate()
    #
    
    #
    
    #
    
    