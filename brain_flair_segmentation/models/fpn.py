from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional
import warnings
warnings.simplefilter("ignore")
 
 

class ConvNormRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, upsample: Optional[bool] = False) -> None:
        super(ConvNormRelu, self).__init__()
        self.upsample = upsample
        self.upsample_block = nn.Upsample(scale_factor= 2, mode= 'bilinear', align_corners= True)
        layers: dict[str, nn.modules] = {
            'conv': nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= 1, padding= 1, bias= False),
            'norm': nn.GroupNorm(32, out_channels),
            'relu': nn.ReLU(inplace= True)
        }
        self.block = nn.Sequential(*layers.values())
    
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.block(x)
        if self.upsample:
            x: torch.Tensor = self.upsample_block(x)
        return x




class SegmentationBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_upsamples: Optional[int] = 0) -> None:
        super(SegmentationBlock, self).__init__()
        layers: dict[str, ConvNormRelu] = {
            'conv_Norm_relu_1': ConvNormRelu(in_channels, out_channels, upsample= bool(n_upsamples))
        }
        if n_upsamples > 1:
            new_layer: dict[str, ConvNormRelu] = {
                f'conv_Norm_relu_{idx + 1}': ConvNormRelu(in_channels, out_channels, upsample= True) for idx in range(1, n_upsamples)
            }
            layers.update(new_layer)
            
        self.block = nn.Sequential(*layers.values())
    
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
        




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
    
         
    
    
    
class Feature_PyramidNetwork(nn.Module):
    def __init__(self, n_classes: Optional[int] = 1, pyramid_channels: Optional[int] = 256, 
                                                     segmentation_channels: Optional[int] = 256) -> None:
        super(Feature_PyramidNetwork, self).__init__()
        self.conv_down1 = DoubleConv(3, 64)
        self.conv_down2 = DoubleConv(64, 128)
        self.conv_down3 = DoubleConv(128, 256)
        self.conv_down4 = DoubleConv(256, 512)        
        self.conv_down5 = DoubleConv(512, 1024)   
        self.maxpool = nn.MaxPool2d(2)
        
        self.toplayer = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0) 
        self.smooth = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.latlayer1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        
        self.seg_blocks = nn.ModuleList([
            SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
            for n_upsamples in [0, 1, 2, 3]
        ])
        
        self.last_conv = nn.Conv2d(256, n_classes, kernel_size=1, stride=1, padding=0)
    
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1: torch.Tensor = self.maxpool(self.conv_down1(x))
        c2: torch.Tensor = self.maxpool(self.conv_down2(c1))
        c3: torch.Tensor = self.maxpool(self.conv_down3(c2))
        c4: torch.Tensor = self.maxpool(self.conv_down4(c3))
        c5: torch.Tensor = self.maxpool(self.conv_down5(c4)) 
        
        p5: torch.Tensor = self.toplayer(c5) 
        p4: torch.Tensor = Feature_PyramidNetwork._upsample_add(p5, self.latlayer1(c4)) 
        p3: torch.Tensor = Feature_PyramidNetwork._upsample_add(p4, self.latlayer2(c3))
        p2: torch.Tensor = Feature_PyramidNetwork._upsample_add(p3, self.latlayer3(c2)) 
        
        p4, p3, p2 = self.smooth(p4), self.smooth(p3), self.smooth(p2)
        
        _, _, h, w = p2.size()
        feature_pyramid: list[torch.Tensor] = [
            seg_block(p) for seg_block, p in zip(self.seg_blocks, [p2, p3, p4, p5])
        ]
        out: torch.Tensor = Feature_PyramidNetwork._upsample(self.last_conv(sum(feature_pyramid)), 4 * h, 4 * w)
        out: torch.Tensor = torch.sigmoid(out)
        return out
    
    
        
    @staticmethod
    def _upsample_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _,_,h,w = y.size()
        upsample = nn.Upsample(size= (h,w), mode= 'bilinear', align_corners= True) 
        return upsample(x) + y
    
    
    @staticmethod
    def _upsample(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        sample = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
        return sample(x)
        
        
    
        
#@: Driver Code
if __name__.__contains__('__main__'):
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fpn = Feature_PyramidNetwork().to(device)
    result: torch.Tensor = fpn(torch.rand(1, 3, 256, 256).to(device))
    print(result.shape)


