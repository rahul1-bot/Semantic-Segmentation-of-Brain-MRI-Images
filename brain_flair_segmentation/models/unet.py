from __future__ import annotations
import torch
import torch.nn as nn
import warnings
warnings.simplefilter("ignore")



class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(DoubleConv, self).__init__()
        layers: dict[str, nn.modules] = {
            'conv_1': nn.Conv2d(in_channels, out_channels, 3, padding= 1),
            'relu_1': nn.ReLU(inplace= True),
            'conv_2': nn.Conv2d(out_channels, out_channels, 3, padding= 1),
            'relu_2': nn.ReLU(inplace= True)
        }
        self.block = nn.Sequential(*layers.values())
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    

    

class UNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(UNet, self).__init__()
        self.conv_down1 = DoubleConv(3, 64)
        self.conv_down2 = DoubleConv(64, 128)
        self.conv_down3 = DoubleConv(128, 256)
        self.conv_down4 = DoubleConv(256, 512)        
        
        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor= 2, mode= 'bilinear', align_corners= True)        
        
        self.conv_up3 = DoubleConv(256 + 512, 256)
        self.conv_up2 = DoubleConv(128 + 256, 128)
        self.conv_up1 = DoubleConv(128 + 64, 64)
        
        self.last_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1: torch.Tensor = self.conv_down1(x) 
        x: torch.Tensor = self.maxpool(conv1)     
        conv2: torch.Tensor = self.conv_down2(x) 
        x: torch.Tensor = self.maxpool(conv2)     
        conv3: torch.Tensor = self.conv_down3(x)  
        x: torch.Tensor = self.maxpool(conv3)     
        x: torch.Tensor = self.conv_down4(x)      
        x: torch.Tensor = self.upsample(x)        
        
        x: torch.Tensor = torch.cat([x, conv3], dim=1) 
        x: torch.Tensor = self.conv_up3(x) 
        x: torch.Tensor = self.upsample(x) 
        x: torch.Tensor = torch.cat([x, conv2], dim=1) 

        x: torch.Tensor = self.conv_up2(x) 
        x: torch.Tensor = self.upsample(x)    
        x: torch.Tensor = torch.cat([x, conv1], dim=1) 
        
        x: torch.Tensor = self.conv_up1(x) 
        
        out: torch.Tensor = self.last_conv(x) 
        out: torch.Tensor = torch.sigmoid(out)
        return out




#@: Driver Code
if __name__.__contains__('__main__'):
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    unet = UNet(n_classes= 1).to(device)
    result: torch.Tensor = unet(torch.rand(1, 3, 256, 256).to(device))
    print(result.shape)