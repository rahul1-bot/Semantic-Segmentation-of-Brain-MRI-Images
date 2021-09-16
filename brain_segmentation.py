from __future__ import annotations

__authors__: list[str] = ['Rahul Sawhney', 'Shubham Garg', 'Harsh Sharma', 'Aabha Malik']

__doc__: str = r'''
    Patent Title: BRAIN MRI SEGMENTATION FOR FLAIR ABNORMALITY DETECTION
    Patent Abstract: ...

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


#@: Custom DataClass
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



#@: -- UNet Architecture -- 
class Double_Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Double_Conv, self).__init__()
        layers: dict[str, object] = {
            'conv_one': nn.Conv2d(in_channels, out_channels, 3, padding= 1),
            'relu_one': nn.ReLU(inplace= True),

            'conv_two': nn.Conv2d(out_channels, out_channels, 3, padding= 1),
            'relu_two': nn.ReLU(inplace= True)
        }
        self.block = nn.Sequential(*layers.values())
    


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


 
class UNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(UNet, self).__init__()
        self.conv_down1 = Double_Conv(3, 64)
        self.conv_down2 = Double_Conv(64, 128)
        self.conv_down3 = Double_Conv(128, 256)
        self.conv_down4 = Double_Conv(256, 512)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor= 2, mode= 'bilinear', align_corners= True)

        self.conv_up3 = Double_Conv(256 + 512, 256)
        self.conv_up2 = Double_Conv(128 + 256, 128)
        self.conv_up1 = Double_Conv(128 + 64, 64)

        self.last_conv = nn.Conv2d(64, n_classes, kernel_size= 1)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv_down1(x)
        x = self.pool(conv1)
        conv2 = self.conv_down2(x)
        x = self.pool(conv2)
        conv3 = self.conv_down3(x)
        x = self.pool(conv3)
        x = self.conv_down4(x)
        x = self.upsample(x)

        x = torch.cat([x, conv3], dim= 1)

        x = self.conv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim= 1)

        x = self.conv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim= 1)

        x = self.conv_up1(x)

        out = self.last_conv(x)
        out = torch.sigmoid(out)

        return out



#@: -- Feature Pyramid Network --
class ConvRelu_Upsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: Optional[bool] = False) -> None:
        super(ConvRelu_Upsample, self).__init__()
        self.upsample = upsample
        self.upsample_layer = nn.Upsample(scale_factor= 2, mode= 'bilinear', align_corners= True)

        layers: dict[str, object] = {
            'conv_one': nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias= False),
            'grp_norm': nn.GroupNorm(32, out_channels),
            'relu_one': nn.ReLU(inplace= True)
        }

        self.block = nn.Sequential(*layers.values())
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.upsample:
            x = self.upsample_layer(x)
        return x



class Segment_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_Upsamples: Optional[int] = 0) -> None:
        super(Segment_Block, self).__init__()
        layers: list[object] = [
            ConvRelu_Upsample(in_channels, out_channels, upsample= bool(n_Upsamples))      
        ]
        if n_Upsamples > 1:
            for _ in range(1, n_Upsamples):
                layers.append(ConvRelu_Upsample(out_channels, out_channels, upsample= True))
        
        self.block = nn.Sequential(*layers)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    


class Feature_Pyramid_Network(nn.Module):
    def __init__(self, n_classes: Optional[int] = 1, pyramid_channels: Optional[int] = 256, 
                                                     segmentation_channels: Optional[int] = 256) -> None:
        super(Feature_Pyramid_Network, self).__init__()
        
        self.conv_down1 = Double_Conv(3, 63)
        self.conv_down2 = Double_Conv(64, 128)
        self.conv_down3 = Double_Conv(128, 256)
        self.conv_down4 = Double_Conv(256, 512)
        self.conv_down5 = Double_Conv(512, 1024)
        self.pool = nn.MaxPool2d(2)

        self.top_layer = nn.Conv2d(1024, 256, kernel_size= 1, stride= 1, padding= 0)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size= 3, stride= 1, padding= 1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size= 3, stride= 1, padding= 1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size= 3, stride= 1, padding= 1)

        self.lateral_1 = nn.Conv2d(512, 256, kernel_size= 1, stride= 1, padding= 0)
        self.lateral_2 = nn.Conv2d(256, 256, kernel_size= 1, stride= 1, padding= 0)
        self.lateral_3 = nn.Conv2d(128, 256, kernel_size= 1, stride= 1, padding= 0)

        self.segment_blocks = nn.ModuleList(
            [Segment_Block(pyramid_channels, segmentation_channels, n_Upsamples= n_Upsamples) for n_Upsamples in [0, 1, 2, 3]]
        )

        self.last_conv = nn.Conv2d(256, n_classes, kernel_size= 1, stride= 1, padding= 0)

    
    
    def upsample_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, H, W = y.size()
        upsample = nn.Upsample(size= (H, W), mode= 'bilinear', align_corners= True)
        return upsample(x) + y
    


    def upsample(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        sample = nn.Upsample(size= (H, W), mode= 'bilinear', align_corners= True)
        return sample(x)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.pool(self.conv_down1(x))
        c2 = self.pool(self.conv_down1(c1))
        c3 = self.pool(self.conv_down1(c2))
        c4 = self.pool(self.conv_down1(c3))
        c5 = self.pool(self.conv_down1(c4))

        p5 = self.top_layer(c5)
        p4 = self.upsample_add(p5, self.lateral_1(c4))
        p3 = self.upsample_add(p4, self.lateral_2(c3))
        p2 = self.upsample_add(p3, self.lateral_1(c2))

        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        _, _, H, W = p2.size()
        feature_pyramid: list[object] = [seg_block(p) for seg_block, p in zip(self.segment_blocks, [p2, p3, p4, p5])]

        out = self.upsample(self.last_conv(sum(feature_pyramid)), H= 4 * H, W= 4 * W)
        out = torch.sigmoid(out)
        return out



#@: -- ResNext_50_UNET -- 
class ConvRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, padding: int) -> None:
        super(ConvRelu, self).__init__()
        layers: dict[str, object] = {
            'conv_one': nn.Conv2d(in_channels, out_channels, kernel_size= kernel, padding= padding),
            'relu_one': nn.ReLU(inplace= True)
        }
        self.block = nn.Sequential(*layers.values())
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)



class Decoder_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Decoder_Block, self).__init__()
        layers: dict[str, object] = {
            'conv_Relu_One': ConvRelu(in_channels, in_channels// 4, 1, 0),
            'de_conv': nn.ConvTranspose2d(in_channels// 4, in_channels// 4, kernel_size= 3, stride= 2, padding= 1, output_padding= 0),
            'conv_Relu_Two': ConvRelu(in_channels// 4, out_channels, 1, 0) 
        }
        self.block = nn.Sequential(*layers.values())
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    


class ResNextUNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(ResNextUNet, self).__init__()
        self.base_model: object = resnext50_32x4d(pretrained= True)
        self.base_layers: list[int] = list(self.base_model.children())
        filters: list[int] = [4*64, 4*128, 4*256, 4*512]

        self.encoder_0 = nn.Sequential(*self.base_layers[:3])
        self.encoder_1 = nn.Sequential(*self.base_layers[4])
        self.encoder_2 = nn.Sequential(*self.base_layers[5])
        self.encoder_3 = nn.Sequential(*self.base_layers[6])
        self.encoder_4 = nn.Sequential(*self.base_layers[7])

        self.decoder_4 = Decoder_Block(filters[3], filters[2])
        self.decoder_3 = Decoder_Block(filters[2], filters[1])
        self.decoder_2 = Decoder_Block(filters[1], filters[0])
        self.decoder_1 = Decoder_Block(filters[0], filters[0])

        self.last_conv_0 = ConvRelu(256, 128, 3, 1)
        self.last_conv_1 = nn.Conv2d(128, n_classes, 3, padding= 1)
    


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_0(x)
        e1 = self.encoder_1(x)
        e2 = self.encoder_2(e1)
        e3 = self.encoder_3(e2)
        e4 = self.encoder_4(e3)

        d4 = self.decoder_4(e4) + e3
        d3 = self.decoder_3(d4) + e2
        d2 = self.decoder_2(d3) + e1
        d1 = self.decoder_1(d2)

        out = self.last_conv_0(d1)
        out = self.last_conv_1(out)
        out = torch.sigmoid(out)

        return out


#@: Segmentation Quality Metrics
class dice_coef_metric(nn.Module):
    def __init__(self) -> None:
        super(dice_coef_metric, self).__init__()
    

    def forward(self, inputs: np.ndarray, target: np.ndarray) -> float:
        intersection = 2.0 * (target * inputs).sum()
        union = target.sum() + inputs.sum()
        if target.sum() == 0 and inputs.sum() == 0:
            return 1.0
        
        return intersection/ union


#@: Segmentation Loss 
__loss_doc__: str = r'''
    LOSS = BCE - Log(DICE)
    where,
        1) BCE = Binary Cross Entropy Loss Function
        2) DICE = Custom Dice Coeffient Loss 

'''

class Custom_Loss(nn.Module):
    def __init__(self) -> None:
        super(Custom_Loss, self).__init__()


    
    def forward(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self._bce_dice_loss(inputs, target)
    


    def _dice_coeff_loss(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth: float = 1.0
        intersection = 2.0 * ((target * inputs).sum()) + smooth
        union = target.sum() + inputs.sum() + smooth
        return 1 - (intersection / union)



    def _bce_dice_loss(self, inputs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_score = self._dice_coeff_loss(inputs, target)
        bce_score = nn.BCELoss()
        bce_loss = bce_score(inputs, target)
        return bce_loss + dice_score



#@: Model Adaptor Class
class Model():
    def __init__(self, model_name: str, net: object, train_loader: object, 
                                                     test_loader: object, 
                                                     criterion: object, 
                                                     optimizer: object, 
                                                     lr_schedular: bool, 
                                                     num_epochs: int, 
                                                     device: torch.device) -> None:
        super(Model, self).__init__()
        self.model_name = model_name
        self.net = net
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = lr_schedular
        self.num_epochs = num_epochs
        self.device = device



    def _compute_IOU(self) -> float:
        threshold: float = 0.3
        test_loss: int| float = 0
    
        with torch.no_grad():
            for i_step, (data, target) in enumerate(self.test_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                
                outputs = self.net(data)

                out_cut = np.copy(outputs.data.cpu().numpy())
                out_cut[np.nonzero( out_cut < threshold )] = 0.0
                out_cut[np.nonzero( out_cut >= threshold )] = 1.0

                pic_loss = dice_coef_metric.forward(out_cut, target.data.cpu().numpy())
                test_loss += pic_loss
        
        return test_loss / i_step 




    def _warmup_lr_schedular(self, warmup_iters: int, warmup_factor: float) -> object:
        def f(x: any) -> float:
            if x >= warmup_iters:
                return 1
            alpha = float(x) / warmup_iters
            return warmup_factor * (1 - alpha) + alpha

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, f)




    def train_validate(self) -> _text:
        print(self.model_name)
        loss_history, self.train_history, self.test_history = [], [], []

        for epoch in range(self.num_epochs):
            self.net.train()
            losses: list = []
            train_iou: list = []

            if self.scheduler:
                warmup_factor = 1.0/ 100
                warmup_iters = min(100, len(self.test_loader) - 1)
                self.scheduler = self._warmup_lr_schedular(warmup_iters, warmup_factor)
            

            for i_step, (data, target) in enumerate(self.train_loader):
                data = data.to(self.device)
                target = target.to(self.device)
                outputs = self.net(data)

                out_cut = np.copy(outputs.data.cpu().numpy())
                out_cut[np.nonzero(out_cut < 0.5)] = 0.0
                out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

                train_dice = dice_coef_metric(out_cut, target.data.cpu().numpy())
                loss = self.criterion(outputs, target)
                
                losses.append(loss.item())
                train_iou.append(train_dice)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if not isinstance(self.scheduler, bool):
                    self.scheduler.step()
                
                val_mean_iou = self._compute_IOU()

                loss_history.append(np.array(losses).mean())
                self.train_history.append(np.array(train_iou).mean())
                self.test_history.append(val_mean_iou)

                print(f'Epoch : {epoch}')
                print(f'Mean Loss on Train : {np.array(losses).mean()}, 
                        Mean DICE on Train : {np.array(train_iou).mean()},
                        Mean DICE on Test : {val_mean_iou}'
                )

        return self.train_history, self.test_history, loss_history

    

    def model_history_plot(self) -> 'plot':
        x = np.arange(self.num_epochs)
        fig = plt.figure(figsize= (10, 6))
        
        plt.plot(x, self.train_history, label= 'Train Dice')
        plt.plot(x, self.test_history, label= 'Test Dice')

        plt.title(f'{self.model_name}', fontsize= 15)
        plt.legend(fontsize= 12)
        plt.xlabel('Epoch', fontsize= 15)
        plt.ylabel('DICE', fontsize= 15)
        plt.show()





#@: driver Code
if __name__.__contains__('__main__'):
    print('hemllo')
    print('Code Still left...')
    