from __future__ import annotations
import torch
import torch.nn as nn
from typing import Optional, Any, Iterable, Union



class ConvNormRelu(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, up_sample: Optional[bool] = False) -> None:
        super(ConvNormRelu, self).__init__()
        self.up_sample = up_sample
        self.add_upSample = nn.Upsample(
            scale_factor= 2, 
            mode= 'bilinear',
            align_corners= True
        )
        layers: dict[str, torch.nn] = {
            'conv': nn.Conv2d(in_channels, out_channels, 3, stride= 1, padding= 1, bias= False),
            'norm': nn.GroupNorm(32, out_channels),
            'relu': nn.ReLU(inplace= True)
        }
        self.block = nn.Sequential(*layers.values())
    
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        if self.up_sample:
            x = self.add_upSample(x)
        return x
    
    


class Segmentation_block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_upSamples: Optional[int] = 0) -> None:
        super(Segmentation_block, self).__init__()
        layers: dict[str, ConvNormRelu] = {
            'conv_Norm_relu_1': ConvNormRelu(in_channels, out_channels, bool(n_upSamples))
        }
        if n_upSamples > 1:
            for idx in range(1, n_upSamples):
                layers.update({f'conv_Norm_relu_{idx + 1}' : ConvNormRelu(out_channels, out_channels, True)})
        
        self.block = nn.Sequential(*layers.values()) 

    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
    
    
    

class Double_Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Double_Conv, self).__init__()
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
    def __init__(self, n_classes: Optional[int] = 1, pyramid_channels: Optional[int] = 256, segmentation_channels: Optional[int] = 256) -> None:
        super(Feature_PyramidNetwork, self).__init__()
        self.down_1 = nn.Sequential(
            Double_Conv(3, 64),
            nn.MaxPool2d(2)
        )
        self.down_2 = nn.Sequential(
            Double_Conv(64, 128),
            nn.MaxPool2d(2)
        )
        self.down_3 = nn.Sequential(
            Double_Conv(128, 256),
            nn.MaxPool2d(2)
        ) 
        self.down_4 = nn.Sequential(
            Double_Conv(256, 512),
            nn.MaxPool2d(2)
        ) 
        self.down_5 = nn.Sequential(
            Double_Conv(512, 1024),
            nn.MaxPool2d(2)
        )
        self.top = nn.Conv2d(1024, 256, kernel_size= 1, stride= 1, padding= 0)
        
        self.smooth1 = nn.Conv2d(256, 256, kernel_size= 3, stride= 1, padding= 1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size= 3, stride= 1, padding= 1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size= 3, stride= 1, padding= 1)
        
        self.lateral_layer_1 = nn.Conv2d(512, 256, kernel_size= 1, stride= 1, padding= 0)
        self.lateral_layer_2 = nn.Conv2d(256, 256, kernel_size= 1, stride= 1, padding= 0)
        self.lateral_layer_3 = nn.Conv2d(128, 256, kernel_size= 1, stride= 1, padding= 0)
        
        seg_layers: dict[str, Segmentation_block] = {
            f'seg_block_{x}': y for x, y in zip(
                [1, 2, 3, 4], [
                    Segmentation_block(pyramid_channels, segmentation_channels, n_upSamples= 0),
                    Segmentation_block(pyramid_channels, segmentation_channels, n_upSamples= 1),
                    Segmentation_block(pyramid_channels, segmentation_channels, n_upSamples= 2),
                    Segmentation_block(pyramid_channels, segmentation_channels, n_upSamples= 3)
                ]
            )
        }
        self.seg_layer_block: list[Segmentation_block] = nn.ModuleList(seg_layers.values())
        self.last_conv = nn.Conv2d(256, n_classes, kernel_size= 1, stride= 1, padding= 0)
        
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down_1(x)
        c2 = self.down_2(c1)
        c3 = self.down_3(c2)
        c4 = self.down_4(c3)
        c5 = self.down_5(c4)
            
        p5 = self.top(c5)
        p4 = Feature_PyramidNetwork._add_upsample(p5, self.lateral_layer_1(c4))
        p3 = Feature_PyramidNetwork._add_upsample(p4, self.lateral_layer_1(c3))
        p2 = Feature_PyramidNetwork._add_upsample(p3, self.lateral_layer_1(c2))
            
        p4, p3, p2 = self.smooth1(p4), self.smooth2(p3), self.smooth3(p2)
            
        _, _, h, w = p2.size()
        feature_list: list[Any] = [
                seg_block(p) for seg_block, p in zip(self.seg_layer_block, [p2, p3, p4, p5])
        ]
        out: torch.Tensor = Feature_PyramidNetwork._upsample(
                x= self.last_conv(sum(feature_list)), 
                h= h * 4,
                w= 4 * w
        )
        out: torch.Tensor = torch.sigmoid(out)
        return out
        
        
        
    @staticmethod
    def _add_upsample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _, _, h, w = y.size()
        upsample = nn.Upsample(size= (h, w), mode= 'bilinear', align_corners= True)
        return upsample(x) + y
        
        
        
    @staticmethod
    def _upsample(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        sample = nn.Upsample(size= (h, w), mode= 'bilinear', align_corners= True)
        return sample(x)