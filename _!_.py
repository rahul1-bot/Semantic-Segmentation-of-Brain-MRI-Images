from __future__ import annotations
__author__: list[str] = ['Rahul Sawhney', 'Leah Khan']

__doc__: str = r'''
    Patent Title: BRAIN MRI SEGMENTATION FOR FLAIR ABNORMALITY DETECTION
    
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
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils



class MRIDataSet(torch.utils.data.Dataset):
    def __init__(self, path: 'dir_path', transform: torchvision.transforms) -> None:
        self.path = path
        self.patients: list[str] = [folder for folder in os.listdir(path) if not folder in ['data.csv', 'README.md']]
        self.transform = transform
        self.masks, self.images = [], []
        
        for patients in self.patients:
            for file in os.listdir(os.path.join(self.path, patients)):
                if 'mask' in file.split('.')[0].split('_'):
                    self.masks.append(os.path.join(self.path, patients, file))
                else:
                    self.images.append(os.path.join(self.path, patients, file))
        
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
    


    def __len__(self) -> int:
        return len(self.images)
    


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        mask = self.masks[index]
        if self.transform is not None:
            image = Image.open(image)
            image = self.transform(image)
            mask = Image.open(mask)
            mask = self.transform(mask)
        return image, mask




class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                                         kernel_size: tuple[int, ...] = (3, 3), 
                                         padding: int = 1) -> None:
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
    def __init__(self, channel1: int, channel2: int, 
                                      kernel_size: tuple[int, ...] = (3, 3), 
                                      padding: int = 1) -> None:
        super(StackEncoder, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size= (2, 2), stride= 2)
        layers: dict[str, object] = {
            'convBlock_1': ConvBlock(channel1, channel2, kernel_size, padding),
            'convBlock_2': ConvBlock(channel1, channel2, kernel_size, padding)
        }
        self.block = nn.Sequential(*layers.values())
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        big_out = self.block(x)
        poolout = self.pool(big_out)
        return big_out, poolout




class StackDecoder(nn.Module):
    def __init__(self, big_channel: int, channel1: int, 
                                         channel2: int, 
                                         kernel_size: tuple[int, ...] = (3, 3), 
                                         padding: int = 1) -> None:
        super(StackDecoder, self).__init__()
        layers: dict[str, object] = {
            'convBlock_1': ConvBlock(channel1 + big_channel, channel2, kernel_size, padding),
            'convBlock_2': ConvBlock(channel2, channel2, kernel_size, padding),
            'convBlock_3': ConvBlock(channel2, channel2, kernel_size, padding)
        }
        self.block = nn.Sequential(*layers.values())
    


    def forward(self, x: torch.Tensor, down_tensor: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = down_tensor.size()
        x = F.upsample(x, size= (height, width), mode= 'bilinear')
        x = torch.cat([x, down_tensor], 1)
        return self.block(x)
    



class UNet(nn.Module):
    def __init__(self, input_shape: int) -> None:
        super(UNet, self).__init__()
        channel, height, width = input_shape

        
        self.stack_encoder_1 = StackEncoder(channel1= channel, channel2= 12, kernel_size= (3, 3))
        self.stack_encoder_2 = StackEncoder(channel1= 12, channel2= 24, kernel_size= (3, 3))
        self.stack_encoder_3 = StackEncoder(channel1= 24, channel2= 46, kernel_size= (3, 3))
        self.stack_encoder_4 = StackEncoder(channel1= 46, channel2= 64, kernel_size= (3, 3))
        self.stack_encoder_5 = StackEncoder(channel1= 64, channel2= 128, kernel_size= (3, 3))
        

        self.centre_layer = ConvBlock(in_channels= 128, out_channels= 128, kernel_size= (3, 3), padding= 1)
        #print(self.down_layers['stack_encoder_1'])
    
        #self.up_layers: dict[str, object] = {
        self.stack_decoder_5 = StackDecoder(big_channel= 128, channel1= 128, channel2= 64, kernel_size= (3, 3))
        self.stack_decoder_4 = StackDecoder(big_channel= 64, channel1= 64, channel2= 46, kernel_size= (3, 3))
        self.stack_decoder_3 = StackDecoder(big_channel= 46, channel1= 46, channel2= 24, kernel_size= (3, 3))
        self.stack_decoder_2 = StackDecoder(big_channel= 24, channel1= 24, channel2= 12, kernel_size= (3, 3))
        self.stack_decoder_1 = StackDecoder(big_channel= 12, channel1= 12, channel2= 12, kernel_size= (3, 3))
        
        self.last_conv = nn.Conv2d(in_channels= 3, out_channels= 1, kernel_size= (1, 1), bias= True)

        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down1, out = self.stack_encoder_1(x)
        down2, out = self.stack_encoder_2(out)
        down3, out = self.stack_encoder_3(out)
        down4, out = self.stack_encoder_4(out)
        down5, out = self.stack_encoder_5(out)

        out = self.centre_layer(out)

        up5 = self.stack_decoder_5(out, down5)
        up4 = self.stack_decoder_4(up5, down4)
        up3 = self.stack_decoder_3(up4, down3)
        up2 = self.stack_decoder_2(up3, down2)
        up1 = self.stack_decoder_1(up2, down1)

        out = self.last_conv(up1)
        return out




class DiceBCELoss(nn.Module):
    def __init__(self, weight: float = None, size_avg: bool = True) -> None:
        super(DiceBCELoss, self).__init__()
    

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        inputs = F.sigmoid(inputs)
        bce_weights: float = 0.5
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth)/ (inputs.sum() + targets.sum() + smooth)
        bce = F.binary_cross_entropy(inputs, targets, reduction= 'mean')
        loss_final = bce * bce_weights + dice_loss * (1 - bce_weights)
        return loss_final  
        


class IOU(nn.Module):
    def __init__(self, weight: float = None, size_avg: bool = True) -> None:
        super(IOU, self).__init__()

    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        iou = (intersection + smooth)/ (union + smooth)
        return iou * 100




class DiceScore(nn.Module):
    def __init__(self, weights: float = None, size_avg: bool = True) -> None:
        super(DiceScore, self).__init__()

    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth: int = 1) -> torch.Tensor:
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice_score = (2. * intersection + smooth)/ (inputs.sum() + targets.sum() + smooth)
        return dice_score





class Model():
    def __init__(self, net: 'model', criterion: object, 
                                     optimizer: object, 
                                     num_epochs: int,
                                     dataloaders: dict[str, object],
                                     device: torch.device) -> None:
        super(Model, self).__init__()
        self.net = net
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.dataloaders = dataloaders
        self.device = device
        


    def train_validate(self) -> dict[str, float]:
        self.train_loss: list[float] = []
        self.val_loss: list[float] = []

        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}/ {self.num_epochs}')
            since: float = time.time()
            running_train_loss: list[float] = []
            
            for images, masks in self.dataloaders['train']:
                images = images.to(self.device)
                masks = masks.to(self.device)
                pred_masks = self.net(images)
                loss = self.criterion(pred_masks, masks)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_train_loss.append(loss.item())
            
            else:
                running_val_loss: list[float] = []
                with torch.no_grad():
                    for images, masks in self.dataloaders['test']:
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        pred_masks = self.net(images)
                        loss = self.criterion(pred_masks, masks)
                        running_val_loss.append(loss.item())
            
            epoch_train_loss = np.mean(running_train_loss)
            print(f'Train Loss: {epoch_train_loss}')
            self.train_loss.append(epoch_train_loss)

            epoch_val_loss = np.mean(running_val_loss)
            print(f'Val Loss: {epoch_val_loss}')
            self.val_loss.append(epoch_val_loss)

            total_time = time.time() - since
            print(f'{total_time// 60:.0f}m {total_time % 60:.0f}s')


    
    def trainVal_lossPlot(self) -> 'plot':
        plt.plot(self.train_loss, label= 'train_loss')
        plt.plot(self.val_loss, label= 'val_loss')
        plt.legend()
        plt.title('Train VS Val Loss Plot')
        plt.show()


    



# driver code
if __name__.__contains__('__main__'):
    path: 'dir_path' = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\brain_segmentation\\kaggle_3m'
    
    transform_list: list[object] = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ]) 

    dataset = MRIDataSet(path, transform_list)
    
    train_data, test_data = torch.utils.data.random_split(dataset, [3600, 329])
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size= 10, shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= 10)

    dataloaders: dict[str, object] = {
        'train': train_loader,
        'test': test_loader
    }
    dataset_sizes: dict[str, int] = {
        'train': len(train_data),
        'test': len(test_data)
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = UNet((3, 256, 256)).to(device)
    
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-3)

    image_segmentor_Model = Model(net= model, criterion= criterion, 
                                              optimizer= optimizer, 
                                              num_epochs= 5, 
                                              dataloaders= dataloaders, 
                                              device= device)
    
    image_segmentor_Model.train_validate()



 

