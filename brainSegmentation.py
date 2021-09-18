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
    Rahul Sawhney: Btech CSE Dept. Amity University, Noida 
    Shubham Garg: Btech CSE Dept. Amity University, Noida
    Harsh Sharma: Btech CSE Dept. Amity University, Noida
    Aabha Malik: Btech CSE Dept. Amity University, Noida

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

__hardware_info__: str = r'''
    CPU: AMD Ryzen 5 3600
    GPU: Nvidia GeForce RTX 2060
    RAM: 32 GB 3600 Mhz
    ROM: Samsung 970 EVO Plus NVME 1 TB

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
from torchvision.models import resnext50_32x4d



#@: Custom DataSet Loader Class 
class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, path: 'dir_path', transform: torchvision.transforms) -> None:
        self.path = path
        self.patients: list[str] = [folder for folder in os.listdir(self.path) if not folder in ['data.csv', 'README.md']]
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
            image, mask = Image.open(image), Image.open(mask)
            image, mask = self.transform(image), self.transform(mask)
        return image, mask



#@: Model 1: UNet   
class UNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(UNet, self).__init__()
        self.conv_down1 = UNet._double_conv(3, 64)
        self.conv_down2 = UNet._double_conv(64, 128)
        self.conv_down3 = UNet._double_conv(128, 256)
        self.conv_down4 = UNet._double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.conv_up3 = UNet._double_conv(256 + 512, 256)
        self.conv_up2 = UNet._double_conv(128 + 256, 128)
        self.conv_up1 = UNet._double_conv(128 + 64, 64)
        
        self.last_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv_down1(x)  
        x = self.maxpool(conv1)     
        conv2 = self.conv_down2(x)  
        x = self.maxpool(conv2)     
        conv3 = self.conv_down3(x)  
        x = self.maxpool(conv3)     
        x = self.conv_down4(x)      
        x = self.upsample(x)        
          
        x = torch.cat([x, conv3], dim=1) 
        
        x = self.conv_up3(x) 
        x = self.upsample(x)  
        x = torch.cat([x, conv2], dim=1) 

        x = self.conv_up2(x)  
        x = self.upsample(x)    
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.conv_up1(x)
        
        out = self.last_conv(x) 
        out = torch.sigmoid(out)
        
        return out
    

    
    @staticmethod
    def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential():
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )




class ConvReluUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                                         upsample: Optional[bool] = False) -> None:
        super(ConvReluUpsample, self).__init__()
        self.upsample = upsample
        self.make_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.upsample:
            x = self.make_upsample(x)
        return x




class SegmentationBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                                         n_upsamples: optional[int] = 0) -> None:
        super(SegmentationBlock, self).__init__()
        blocks: list[object] = [
            ConvReluUpsample(in_channels, out_channels, upsample=bool(n_upsamples))
        ]
        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(ConvReluUpsample(out_channels, out_channels, upsample=True))
        self.block = nn.Sequential(*blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)





#@: Model 2: Convolutional Feature Pyramid network 
class FeaturePyramid_Network(nn.Module):
    def __init__(self, n_classes: Optional[int] = 1, pyramid_channels: Optional[int] = 256, 
                                                     segmentation_channels: Optional[int] = 256) -> None:
        super(FeaturePyramid_Network, self).__init__()
        self.conv_down1 = FeaturePyramid_Network._double_conv(3, 64)
        self.conv_down2 = FeaturePyramid_Network._double_conv(64, 128)
        self.conv_down3 = FeaturePyramid_Network._double_conv(128, 256)
        self.conv_down4 = FeaturePyramid_Network._double_conv(256, 512)        
        self.conv_down5 = FeaturePyramid_Network._double_conv(512, 1024)   
        self.maxpool = nn.MaxPool2d(2)
        
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [0, 1, 2, 3]
        ])
        
        self.last_conv = nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0)
        


    def upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _,_,H,W = y.size()
        upsample = nn.Upsample(size=(H,W), mode='bilinear', align_corners=True) 
        return upsample(x) + y
    


    def upsample(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        sample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
        return sample(x)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.maxpool(self.conv_down1(x))
        c2 = self.maxpool(self.conv_down2(c1))
        c3 = self.maxpool(self.conv_down3(c2))
        c4 = self.maxpool(self.conv_down4(c3))
        c5 = self.maxpool(self.conv_down5(c4)) 
        
        p5 = self.toplayer(c5) 
        p4 = self.upsample_add(p5, self.latlayer1(c4)) 
        p3 = self.upsample_add(p4, self.latlayer2(c3))
        p2 = self.upsample_add(p3, self.latlayer3(c2)) 
        
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)
        
        _, _, h, w = p2.size()
        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p2, p3, p4, p5])]
        
        out = self.upsample(self.last_conv(sum(feature_pyramid)), 4 * h, 4 * w)
        
        out = torch.sigmoid(out)
        return out

    

    @staticmethod
    def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential():
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )




class ConvRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                                         kernel: int, 
                                         padding: int) -> None:
        super(ConvRelu, self).__init__()
        self.convrelu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
            nn.ReLU(inplace=True)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor :
        x = self.convrelu(x)
        return x




class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DecoderBlock, self).__init__()
        self.conv1 = ConvRelu(in_channels, in_channels // 4, 1, 0)
        self.deconv = nn.ConvTranspose2d(
                              in_channels // 4, 
                              in_channels // 4, 
                              kernel_size=4,
                              stride=2, 
                              padding=1, 
                              output_padding=0
                            )
        self.conv2 = ConvRelu(in_channels // 4, out_channels, 1, 0)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x



#@: Model 3: UNet with ResNext-50
class ResNeXtUNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(ResNeXtUNet, self).__init__()
        self.base_model = resnext50_32x4d(pretrained=True)
        self.base_layers = list(self.base_model.children())
        filters: list[int] = [4*64, 4*128, 4*256, 4*512]
        
        self.encoder0 = nn.Sequential(*self.base_layers[:3])
        self.encoder1 = nn.Sequential(*self.base_layers[4])
        self.encoder2 = nn.Sequential(*self.base_layers[5])
        self.encoder3 = nn.Sequential(*self.base_layers[6])
        self.encoder4 = nn.Sequential(*self.base_layers[7])

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.last_conv0 = ConvRelu(256, 128, 3, 1)
        self.last_conv1 = nn.Conv2d(128, n_classes, 3, padding=1)
                       


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder0(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
    
        out = self.last_conv0(d1)
        out = self.last_conv1(out)
        out = torch.sigmoid(out)    
        return out




#@: Custom Metric Class: 
class Metric(nn.Module):
    def __init__(self, inputs: np.ndarray, target: np.ndarray) -> None:
        super(Metric, self).__init__()
        self.inputs = inputs
        self.target = target


    def forward(self) -> float:
        intersection = 2.0 * (self.target * self.inputs).sum()
        union = self.target.sum() + self.inputs.sum()
        if self.target.sum() == 0 and self.inputs.sum() == 0:
            return 1.0
        return intersection / union




#@: Custom Loss Class 
class Loss(nn.Module):
    def __init__(self, inputs: torch.Tensor, target: torch.Tensor) -> None:
        super(Loss, self).__init__()
        self.inputs = inputs
        self.target = target
    

    def forward(self) -> torch.Tensor:
        return Loss.bce_dice_loss(self.inputs, self.target)
    


    @staticmethod
    def dice_coef_loss(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth = 1.0
        intersection = 2.0 * ((target * inputs).sum()) + smooth
        union = target.sum() + inputs.sum() + smooth
        return 1 - (intersection / union)



    @staticmethod
    def bce_dice_loss(inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dicescore = Loss.dice_coef_loss(inputs, target)
        bcescore = nn.BCELoss()
        bceloss = bcescore(inputs, target)
        return bceloss + dicescore
    


#@: Model Adaptor Class 
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
        self.lr_scheduler = lr_scheduler



    def train_validate(self, history: Optional[bool] = False) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        print(self.model_name)
        loss_history: list = []
        train_history: list = []
        val_history: list = []

        for epoch in range(self.num_epochs):
            self.model.train()
            losses: list = []
            train_iou: list = []
                    
            if self.lr_scheduler:
                warmup_factor: float = 1.0 / 100
                warmup_iters: int = min(100, len(self.train_loader) - 1)
                self.lr_scheduler = self.__warmup_lr_scheduler(warmup_iters, warmup_factor)
            
            for i_step, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                outputs = self.model(data)
                
                out_cut = np.copy(outputs.data.cpu().numpy())
                out_cut[np.nonzero(out_cut < 0.5)] = 0.0
                out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
                
                train_dice = self.metric(out_cut, target.data.cpu().numpy()).forward()
                loss = self.criterion(outputs, target).forward()
                
                losses.append(loss.item())
                train_iou.append(train_dice)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                if self.lr_scheduler:
                    self.lr_scheduler.step()
    
            val_mean_iou = self.__compute_iou()
            
            loss_history.append(np.array(losses).mean())
            train_history.append(np.array(train_iou).mean())
            val_history.append(val_mean_iou)
            
            print("Epoch [%d]" % (epoch))
            print("Mean loss on train:", np.array(losses).mean(), 
                  "\nMean DICE on train:", np.array(train_iou).mean(), 
                  "\nMean DICE on validation:", val_mean_iou
            )
        
        if history:
            return loss_history, train_history, val_history
    


    def __compute_iou(self, threshold: Optional[float] = 0.3) -> float:
        valloss: float = 0
        with torch.no_grad():
            for i_step, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                outputs = self.model(data)
            
                out_cut = np.copy(outputs.data.cpu().numpy())
                out_cut[np.nonzero(out_cut < threshold)] = 0.0
                out_cut[np.nonzero(out_cut >= threshold)] = 1.0

                picloss = self.metric(out_cut, target.data.cpu().numpy()).forward()
                valloss += picloss

        return valloss / i_step




    def __warmup_lr_scheduler(self, warmup_iters: int, warmup_factor: float) -> torch.optim.lr_scheduler.LRScheduler():
        def f(x: int) -> int| float:
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, f)





#@: Driver Code 
if __name__.__contains__('__main__'):
    path: 'dir_path' = 'C:\\Users\\RAHUL\\OneDrive\\Desktop\\brain_segmentation\\kaggle_3m'
    
    transform_list: list[object] = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor()
    ]) 
    dataset = BrainDataset(path, transform_list)
    for image, mask in dataset:
        print(image.shape)
        print(mask.shape)
        break
    
    train_data, test_data = torch.utils.data.random_split(dataset, [3600, 329])
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size= 16, shuffle= True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size= 16)

    dataloaders: dict[str, object] = {
        'train': train_loader,
        'test': test_loader
    }
    dataset_sizes: dict[str, int] = {
        'train': len(train_data),
        'test': len(test_data)
    }

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # models 
    unet = UNet(n_classes= 1).to(device)
    fpn = FeaturePyramid_Network().to(device)
    rx50 = ResNeXtUNet(n_classes= 1).to(device)
    
    # Models Optimizers
    unet_optimizer = torch.optim.Adamax(unet.parameters(), lr=1e-3)
    fpn_optimizer = torch.optim.Adamax(fpn.parameters(), lr=1e-3)
    rx50_optimizer = torch.optim.Adam(rx50.parameters(), lr=5e-4)
  
    # Image Segmentor Models
    unet_model = Model(
        model_name= 'UNet',
        model= unet,
        train_loader= train_loader,
        test_loader= test_loader,
        criterion= Loss,
        optimizer= unet_optimizer,
        num_epochs= 25,
        metric= Metric,
        device= device,
        lr_scheduler= False
    )

    fpn_model = Model(
        model_name= 'Convolutional Feature Pyramid Network',
        model= fpn,
        train_loader= train_loader,
        test_loader= test_loader,
        criterion= criterion,
        optimizer= fpn_optimizer,
        num_epochs= 25,
        device= device,
        lr_scheduler= False # True
    )

    resnext_model = Model(
        model_name= "ResNeXt50",
        model= rx50,
        train_loader= train_loader,
        test_loader= test_loader,
        criterion= Loss,
        optimizer= rx50_optimizer,
        num_epochs= 25,
        metric= Metric,
        device= device,
        lr_scheduler= False
    )

    # run 
    unet_model.train_validate()
    fpn_model.train_validate()
    resnext_model.train_validate()

    
    
    